# IF-DreamFusion

This is a pytorch implementation of the text-to-3D model **Dreamfusion**, powered by [DeepFloyd-IF](https://github.com/deep-floyd/IF).

This repository is a hard clone of [Stable-Dreamfusion](https://github.com/ashawkey/stable-dreamfusion), and the only difference is the integration of DeepFloyd-IF.

<p float="left">
  <img src="https://user-images.githubusercontent.com/5498512/236260854-e596e67e-d9d3-4355-8cb4-7f2d4f4e1eff.gif">
  <img src="https://user-images.githubusercontent.com/5498512/236260261-383d5339-c6e9-47ba-b15d-18020148a2df.gif">
</p>

## 🔥TODO
- [ ] Optimize for IF Diffusion Model
- [ ] Release Colab Demo

## ⚡️Comparison w/ Stable-DreamFusion

| | **IF-DreamFusion** | Stable-DreamFusion |
| ------------------ | :------------------: | :------------------: |
| Compositional Scene | V | |
| Geometry | V | |
| Resolution | | V |

# Install

```bash
git clone https://github.com/SusungHong/IF-DreamFusion.git
cd IF-DreamFusion
```

To use image-conditioned 3D generation, you need to download some pretrained checkpoints manually:
* [Zero-1-to-3](https://github.com/cvlab-columbia/zero123) for diffusion backend.
    We use `105000.ckpt` by default, and it is hard-coded in `guidance/zero123_utils.py`.
    ```bash
    cd pretrained/zero123
    wget https://huggingface.co/cvlab/zero123-weights/resolve/main/105000.ckpt
    ```
* [Omnidata](https://github.com/EPFL-VILAB/omnidata/tree/main/omnidata_tools/torch) for depth and normal prediction. 
    These ckpts are hardcoded in `preprocess_image.py`.
    ```bash
    cd pretrained/omnidata
    # assume gdown is installed
    gdown '1Jrh-bRnJEjyMCS7f-WsaFlccfPjJPPHI&confirm=t' # omnidata_dpt_depth_v2.ckpt
    gdown '1wNxVO4vVbDEMEpnAi_jwQObf2MFodcBR&confirm=t' # omnidata_dpt_normal_v2.ckpt
    ```

### Install with pip
```bash
pip install -r requirements.txt
```

### Build extension (optional)
By default, we use [`load`](https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.load) to build the extension at runtime.
We also provide the `setup.py` to build each extension:
```bash
# install all extension modules
bash scripts/install_ext.sh

# if you want to install manually, here is an example:
pip install ./raymarching # install to python path (you still need the raymarching/ folder, since this only installs the built extension.)
```

### Taichi backend (optional)
Use [Taichi](https://github.com/taichi-dev/taichi) backend for Instant-NGP. It achieves comparable performance to CUDA implementation while **No CUDA** build is required. Install Taichi with pip:
```bash
pip install -i https://pypi.taichi.graphics/simple/ taichi-nightly
```

### Trouble Shooting:
* we assume working with the latest version of all dependencies, if you meet any problems from a specific dependency, please try to upgrade it first (e.g., `pip install -U diffusers`). If the problem still holds, [reporting a bug issue](https://github.com/ashawkey/stable-dreamfusion/issues/new?assignees=&labels=bug&template=bug_report.yaml&title=%3Ctitle%3E) will be appreciated!
* `[F glutil.cpp:338] eglInitialize() failed Aborted (core dumped)`: this usually indicates problems in OpenGL installation. Try to re-install Nvidia driver, or use nvidia-docker as suggested in https://github.com/ashawkey/stable-dreamfusion/issues/131 if you are using a headless server.
* `TypeError: xxx_forward(): incompatible function arguments`： this happens when we update the CUDA source and you used `setup.py` to install the extensions earlier. Try to re-install the corresponding extension (e.g., `pip install ./gridencoder`).

### Tested environments
* Ubuntu 22 with torch 1.12 & CUDA 11.6 on a V100.

# Usage

First time running will take some time to compile the CUDA extensions.

```bash
#### if-dreamfusion setting

### Instant-NGP NeRF Backbone
# + faster rendering speed
# + less GPU memory (~16G)
# - need to build CUDA extensions (a CUDA-free Taichi backend is available)

## train with text prompt (with the default settings)
# `-O` equals `--cuda_ray --fp16`
# `--cuda_ray` enables instant-ngp-like occupancy grid based acceleration.
python main.py --text "a hamburger" --workspace trial -O

# reduce if-diffusion memory usage with `--vram_O` 
# enable various vram savings (https://huggingface.co/docs/diffusers/optimization/fp16).
# note that the vram settings are disabled in IF-DreamFusion
python main.py --text "a hamburger" --workspace trial -O --vram_O
# this makes it possible to train with larger rendering resolution, which leads to better quality (see https://github.com/ashawkey/stable-dreamfusion/pull/174)
python main.py --text "a hamburger" --workspace trial -O --vram_O --w 300 --h 300 # Tested to run fine on 8GB VRAM (Nvidia 3070 Ti).

# use CUDA-free Taichi backend with `--backbone grid_taichi`
python3 main.py --text "a hamburger" --workspace trial -O --backbone grid_taichi

# choose if-diffusion version (support 1.0, default is 1.0 now)
python main.py --text "a hamburger" --workspace trial -O --if_version 1.0

# we also support negative text prompt now:
python main.py --text "a rose" --negative "red" --workspace trial -O

# use original Stable Diffusion (default is DeepFloyd-IF)
python main.py --text "a rose" --negative "red" --workspace trial -O --guidance stable-diffusion

# A Gradio GUI is also possible (with less options):
python gradio_app.py # open in web browser

## after the training is finished:
# test (exporting 360 degree video)
python main.py --workspace trial -O --test
# also save a mesh (with obj, mtl, and png texture)
python main.py --workspace trial -O --test --save_mesh
# test with a GUI (free view control!)
python main.py --workspace trial -O --test --gui

### Vanilla NeRF backbone
# + pure pytorch, no need to build extensions!
# - slow rendering speed
# - more GPU memory

## train
# `-O2` equals `--backbone vanilla`
python main.py --text "a hotdog" --workspace trial2 -O2

# if CUDA OOM, try to reduce NeRF sampling steps (--num_steps and --upsample_steps)
python main.py --text "a hotdog" --workspace trial2 -O2 --num_steps 64 --upsample_steps 0

## test
python main.py --workspace trial2 -O2 --test
python main.py --workspace trial2 -O2 --test --save_mesh
python main.py --workspace trial2 -O2 --test --gui # not recommended, FPS will be low.

### DMTet finetuning

## use --dmtet and --init_ckpt <nerf checkpoint> to finetune the mesh at higher reslution
python main.py -O --text "a hamburger" --workspace trial_dmtet --dmtet --iters 5000 --init_ckpt trial/checkpoints/df.pth

## test & export the mesh
python main.py -O --text "a hamburger" --workspace trial_dmtet --dmtet --iters 5000 --test --save_mesh

## gui to visualize dmtet
python main.py -O --text "a hamburger" --workspace trial_dmtet --dmtet --iters 5000 --test --gui

### Image-conditioned 3D Generation

## preprocess input image
# note: the results of image-to-3D is dependent on zero-1-to-3's capability. For best performance, the input image should contain a single front-facing object. Check the examples under ./data.
# this will exports `<image>_rgba.png`, `<image>_depth.png`, and `<image>_normal.png` to the directory containing the input image.
python preprocess_image.py <image>.png 

## train
# pass in the processed <image>_rgba.png by --image and do NOT pass in --text to enable zero-1-to-3 backend.
python main.py -O --image <image>_rgba.png --workspace trial_image --iters 5000
# dmtet finetuning (highly recommended)
python main.py -O --image <image>_rgba.png --workspace trial_image_dmtet --dmtet --init_ckpt trial_image/checkpoints/df.pth

# experimental: providing both --text and --image enables stable-diffusion backend, but the result may look very different from the provided image. This is still an option if image-only mode cannot produce a satisfactory result.
python main.py -O --image hamburger_rgba.png --text "a DSLR photo of a delicious hamburger" --workspace trial_image_text

## test / visualize
python main.py -O --image <image>_rgba.png --workspace trial_image_dmtet --dmtet --test --save_mesh
python main.py -O --image <image>_rgba.png --workspace trial_image_dmtet --dmtet --test --gui
```

For advanced tips and other developing stuff, check [Advanced Tips](./assets/advanced.md).

# Acknowledgement

This work is based on an increasing list of amazing research works and open-source projects, thanks a lot to all the authors for sharing!

* [DreamFusion: Text-to-3D using 2D Diffusion](https://dreamfusion3d.github.io/)
    ```
    @article{poole2022dreamfusion,
        author = {Poole, Ben and Jain, Ajay and Barron, Jonathan T. and Mildenhall, Ben},
        title = {DreamFusion: Text-to-3D using 2D Diffusion},
        journal = {arXiv},
        year = {2022},
    }
    ```

* [Magic3D: High-Resolution Text-to-3D Content Creation](https://research.nvidia.com/labs/dir/magic3d/)
   ```
   @inproceedings{lin2023magic3d,
      title={Magic3D: High-Resolution Text-to-3D Content Creation},
      author={Lin, Chen-Hsuan and Gao, Jun and Tang, Luming and Takikawa, Towaki and Zeng, Xiaohui and Huang, Xun and Kreis, Karsten and Fidler, Sanja and Liu, Ming-Yu and Lin, Tsung-Yi},
      booktitle={IEEE Conference on Computer Vision and Pattern Recognition ({CVPR})},
      year={2023}
    }
   ```

* [Zero-1-to-3: Zero-shot One Image to 3D Object](https://github.com/cvlab-columbia/zero123)
    ```
    @misc{liu2023zero1to3,
        title={Zero-1-to-3: Zero-shot One Image to 3D Object}, 
        author={Ruoshi Liu and Rundi Wu and Basile Van Hoorick and Pavel Tokmakov and Sergey Zakharov and Carl Vondrick},
        year={2023},
        eprint={2303.11328},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
    }
    ```

* [RealFusion: 360° Reconstruction of Any Object from a Single Image](https://github.com/lukemelas/realfusion)
    ```
    @inproceedings{melaskyriazi2023realfusion,
        author = {Melas-Kyriazi, Luke and Rupprecht, Christian and Laina, Iro and Vedaldi, Andrea},
        title = {RealFusion: 360 Reconstruction of Any Object from a Single Image},
        booktitle={CVPR}
        year = {2023},
        url = {https://arxiv.org/abs/2302.10663},
    }
    ```

* [Fantasia3D: Disentangling Geometry and Appearance for High-quality Text-to-3D Content Creation](https://fantasia3d.github.io/)
    ```
    @article{chen2023fantasia3d,
        title={Fantasia3D: Disentangling Geometry and Appearance for High-quality Text-to-3D Content Creation},
        author={Rui Chen and Yongwei Chen and Ningxin Jiao and Kui Jia},
        journal={arXiv preprint arXiv:2303.13873},
        year={2023}
    }
    ```

* [Stable Diffusion](https://github.com/CompVis/stable-diffusion) and the [diffusers](https://github.com/huggingface/diffusers) library.

    ```
    @misc{rombach2021highresolution,
        title={High-Resolution Image Synthesis with Latent Diffusion Models},
        author={Robin Rombach and Andreas Blattmann and Dominik Lorenz and Patrick Esser and Björn Ommer},
        year={2021},
        eprint={2112.10752},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
    }

    @misc{von-platen-etal-2022-diffusers,
        author = {Patrick von Platen and Suraj Patil and Anton Lozhkov and Pedro Cuenca and Nathan Lambert and Kashif Rasul and Mishig Davaadorj and Thomas Wolf},
        title = {Diffusers: State-of-the-art diffusion models},
        year = {2022},
        publisher = {GitHub},
        journal = {GitHub repository},
        howpublished = {\url{https://github.com/huggingface/diffusers}}
    }
    ```

* The GUI is developed with [DearPyGui](https://github.com/hoffstadt/DearPyGui).
