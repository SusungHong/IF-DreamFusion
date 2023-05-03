from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler, IFPipeline
from diffusers.utils.import_utils import is_xformers_available
from os.path import isfile

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import custom_bwd, custom_fwd
from dataclasses import dataclass


class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

@dataclass
class UNet2DConditionOutput:
    sample: torch.HalfTensor # Not sure how to check what unet_traced.pt contains, and user wants. HalfTensor or FloatTensor

class DeepFloyd_IF(nn.Module):
    def __init__(self, device, fp16, vram_O, if_version='1.0', hf_key=None, t_range=[0.02, 0.98]):
        super().__init__()

        self.device = device
        self.if_version = if_version

        print(f'[INFO] loading stable diffusion...')

        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif self.if_version == '1.0':
            model_key = "DeepFloyd/IF-I-XL-v1.0"
        else:
            raise ValueError(f'DeepFloyd-IF version {self.if_version} not supported.')

        self.precision_t = torch.float16 if fp16 else torch.float32

        # Create model
        pipe = IFPipeline.from_pretrained(model_key, variant="fp16", torch_dtype=torch.float16)
        pipe.enable_model_cpu_offload()

        if isfile('./unet_traced.pt'):
            # use jitted unet
            unet_traced = torch.jit.load('./unet_traced.pt')
            class TracedUNet(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.in_channels = pipe.unet.in_channels
                    self.device = pipe.unet.device

                def forward(self, latent_model_input, t, encoder_hidden_states):
                    sample = unet_traced(latent_model_input, t, encoder_hidden_states)[0]
                    return UNet2DConditionOutput(sample=sample)
            pipe.unet = TracedUNet()

        # if vram_O:
        #     pipe.enable_sequential_cpu_offload()
        #     pipe.unet.to(memory_format=torch.channels_last)
        #     pipe.enable_attention_slicing(1)
        #     # pipe.enable_model_cpu_offload()
        # else:
        #     pipe.to(device)

        self.pipe = pipe
        
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler", torch_dtype=self.precision_t)

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        print(f'[INFO] loaded DeepFloyd-IF!')

    @torch.no_grad()
    def get_text_embeds(self, prompt):

        device = self.pipe._execution_device

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # while T5 can handle much longer input sequences than 77, the text encoder was trained with a max length of 77 for IF
        max_length = 77

        prompt = self.pipe._text_preprocessing(prompt, clean_caption=False)
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_length - 1 : -1])

        attention_mask = text_inputs.attention_mask.to(device)

        prompt_embeds = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        prompt_embeds = prompt_embeds[0]

        if self.text_encoder is not None:
            dtype = self.text_encoder.dtype
        elif self.unet is not None:
            dtype = self.unet.dtype
        else:
            dtype = None

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, 1, 1)
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)

        return prompt_embeds


    def train_step(self, text_embeddings, pred_rgb, guidance_scale=100, as_latent=False, grad_scale=1):
        
        if as_latent:
            latents = F.interpolate(pred_rgb, (64, 64), mode='bilinear', align_corners=False) * 2 - 1
        else:
            pred_rgb_64 = F.interpolate(pred_rgb, (64, 64), mode='bilinear', align_corners=False)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(pred_rgb_64)
            pred_rgb_noisy = self.scheduler.add_noise(pred_rgb_64, noise, t)
            
            model_input = torch.cat([pred_rgb_noisy] * 2)
            model_input = self.scheduler.scale_model_input(model_input, t)

            noise_pred = self.unet(model_input, t, encoder_hidden_states=text_embeddings, cross_attention_kwargs=None).sample

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
        noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1], dim=1)
        noise_pred_pos, predicted_variance = noise_pred_pos.split(model_input.shape[1], dim=1)

        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)
        

        # import kiui
        # latents_tmp = torch.randn((1, 4, 64, 64), device=self.device)
        # latents_tmp = latents_tmp.detach()
        # kiui.lo(latents_tmp)
        # self.scheduler.set_timesteps(30)
        # for i, t in enumerate(self.scheduler.timesteps):
        #     latent_model_input = torch.cat([latents_tmp] * 3)
        #     noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']
        #     noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
        #     noise_pred = noise_pred_uncond + 10 * (noise_pred_pos - noise_pred_uncond)
        #     latents_tmp = self.scheduler.step(noise_pred, t, latents_tmp)['prev_sample']
        # imgs = self.decode_latents(latents_tmp)
        # kiui.vis.plot_image(imgs)

        # w(t), sigma_t^2
        w = (1 - self.alphas[t])
        grad = grad_scale * w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        # since we omitted an item in grad, we need to use the custom function to specify the gradient
        loss = SpecifyGradient.apply(pred_rgb_64, grad)

        return loss

    @torch.no_grad()
    def produce_latents(self, text_embeddings, height=64, width=64, num_inference_steps=100, guidance_scale=7.5, latents=None):

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8), device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            # Save input tensors for UNet
            #torch.save(latent_model_input, "produce_latents_latent_model_input.pt")
            #torch.save(t, "produce_latents_t.pt")
            #torch.save(text_embeddings, "produce_latents_text_embeddings.pt")
            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

            # perform guidance
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
        
        return latents

    def decode_latents(self, imgs):

        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        return imgs


if __name__ == '__main__':

    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', type=str)
    parser.add_argument('--negative', default='', type=str)
    parser.add_argument('--if_version', type=str, default='1.0', choices=['1.0'], help="IF version")
    parser.add_argument('--hf_key', type=str, default=None, help="hugging face Stable diffusion model key")
    parser.add_argument('--fp16', action='store_true', help="use float16 for training")
    parser.add_argument('--vram_O', action='store_true', help="optimization for low VRAM usage")
    parser.add_argument('-H', type=int, default=64)
    parser.add_argument('-W', type=int, default=64)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device('cuda')

    deepfloyd_if = DeepFloyd_IF(device, opt.fp16, opt.vram_O, opt.if_version, opt.hf_key)

    imgs = deepfloyd_if.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)

    # visualize image
    plt.imshow(imgs[0])
    plt.show()




