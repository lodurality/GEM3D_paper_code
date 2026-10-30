# GEM3D: Generative Medial Abstractions for 3D Shape Synthesis

### [Project Page](https://lodurality.github.io/GEM3D/) | [Paper (arXiv)](https://arxiv.org/abs/2402.16994)

**Official PyTorch implementation of GEM3D (SIGGRAPH 2024)**

## UPDATES

- **2026-10-30:** Added a checkpoint guide covering file layout, model roles, and checkpoint formats.
- **2026-10-30:** Updated the Google Drive checkpoint download link after the previous link was disabled by UMass.

## Quickstart
Download model checkpoints [here](https://drive.google.com/drive/folders/1CBml8E9mUek4-eHeGLVKFRnGavFCv11t). Place them in the `submission_checkpoints` folder in the root of the repository. See [Model checkpoints](#model-checkpoints) for the expected files and formats.

### Generation
To generate mesh using GEM3D run the following command
```
bash ./bash_scripts/generate_mesh.sh <category-id> <skeleton_seed> <latent_seed>
```
For example, to generate a chair run the following:
```
bash ./bash_scripts/generate_mesh.sh 03001627 0 0
```
Results will be saved in `custom_outputs/generation/` folder.  We provide some examples in `sample_outputs/generation/` folder for reference

### Reconstruction
To reconstruct a single mesh run the following:
```
python reconstruct_single_mesh.py \
	--input_mesh_path=sample_data/single_mesh/bucky60.stl \
	--output_folder=custom_outputs/reconstruction/single_mesh \
	--model_path=submission_checkpoints/skelnet_skelrays_reg_abs_loss_min_sdf_1000_rays_8_heads.pth \
	--skelmodel_path=submission_checkpoints/p2p_vecset_skeletons_min_sdf_iter_50_p2p_min_sdf_vecset_from_ckpt.pth \
	--vis_outputs
```
To run ShapeNet reconstruction on sample data run the following:
```
bash ./bash_scripts/reconstruct_shapenet.sh
```
To run Thingi10K reconstruction using sample data run the following:
```
bash ./bash_scripts/reconstruct_thingi10k.sh
```
Outputs will be saved in `custom_outputs/reconstruction/` folder.  We provide sample outputs of running the code in `sample_outputs/reconstruction/` folder.

## Model checkpoints

GEM3D uses separate models for skeleton prediction, shape reconstruction, and the two diffusion stages. The guide below maps the checkpoint paths used by the supplied scripts to their roles, so you can select the files needed for inference or training.

### Expected file layout

The scripts reference the following files under the repository root:

```text
submission_checkpoints/
├── skelnet_skelrays_reg_abs_loss_min_sdf_1000_rays_8_heads.pth
├── skelnet_skelrays_reg_abs_loss_min_sdf_1000_rays_8_heads_1200.pth
├── p2p_vecset_skeletons_min_sdf_iter_50_p2p_min_sdf_vecset_from_ckpt.pth
├── checkpoint_skel_ldm_1024.pth
├── checkpoint_l256_fin.pth
└── kl_ae_skeletons_min_sdf_iter_50_dim_256_2080ti.pth
```

| Checkpoint | Role and use |
| --- | --- |
| `skelnet_skelrays_reg_abs_loss_min_sdf_1000_rays_8_heads.pth` | Skeleton autoencoder with the envelope-based implicit decoder. Used by reconstruction (`--model_path`) and as the pretrained autoencoder for latent VAE and surface diffusion training. |
| `skelnet_skelrays_reg_abs_loss_min_sdf_1000_rays_8_heads_1200.pth` | Skeleton autoencoder checkpoint selected by `bash_scripts/generate_shapes.sh` (`--ae-pth`) to decode generated surface latents into a mesh. |
| `p2p_vecset_skeletons_min_sdf_iter_50_p2p_min_sdf_vecset_from_ckpt.pth` | Skeleton prediction model for reconstruction (`--skelmodel_path`). The reconstruction code loads this into a separate model and copies its `skel_model` into the reconstruction autoencoder. |
| `checkpoint_skel_ldm_1024.pth` | Shape-label-conditioned skeleton diffusion model (`--skel_dm_pth`). Samples skeleton point clouds. |
| `checkpoint_l256_fin.pth` | Skeleton- and shape-label-conditioned surface diffusion model (`--latent_dm_pth`), including its latent VAE. The generation script uses `--kl_latent_dim 256`. |
| `kl_ae_skeletons_min_sdf_iter_50_dim_256_2080ti.pth` | Standalone latent VAE used to initialize `--latent_encoder_path` during surface diffusion training. Maps autoencoder features to and from the 256-dimensional diffusion latent space. |

For **reconstruction**, use the autoencoder without the `_1200` suffix and the skeleton prediction checkpoint. For **full mesh generation**, `bash_scripts/generate_shapes.sh` uses the `_1200` autoencoder and both diffusion checkpoints. Generation loads the latent VAE from the surface diffusion checkpoint, so it does not load the standalone VAE file separately.

These filenames describe the paths expected by the code. Adjust the paths in the shell scripts to your local checkout and downloaded filenames; the scripts currently use `/home/$USER/Repos/gem3d_paper_code/`.

### Checkpoint contents

- **Autoencoder, skeleton prediction, and standalone latent VAE:** plain PyTorch state dictionaries, saved with `torch.save(model.state_dict(), path)` and loaded directly with `model.load_state_dict(torch.load(path))`. There is no outer `model` key. Reconstruction uses `strict=False` when loading the autoencoder and skeleton prediction weights.
- **Skeleton and surface diffusion:** dictionaries whose `model` entry is the model state dictionary, loaded with `model.load_state_dict(torch.load(path)['model'])`. The surface diffusion model includes a `latent_encoder` submodule, so its state dictionary also contains the VAE weights under `latent_encoder.*`.

The diffusion training save helpers in `utils_vecset/misc.py` additionally store `optimizer`, `epoch`, `scaler`, and `args` for resuming training. They write `checkpoint-<epoch>.pth` and `checkpoint_current.pth` under the configured output directory. Inference reads only `model`; the other three training scripts save plain state dictionaries without optimizer or epoch metadata.

## Training

Since our dataset is fairly large we provide commands to run training on sample data. For full scale training download our data and replace links accordingly. Download instructions and data structure description are provided [below](https://github.com/lodurality/GEM3D_paper_code?tab=readme-ov-file#preprocessed-data). 

### Shape reconstruction
Reconstruction training consists of two independent stages: skeleton autoencoder training and skeleton prediction training (for inference).

To train skeleton autoencoder with envelope-based implicit function run the following:
```
bash ./bash_scripts/train_skeleton_autoencoder.sh 
```
To train skeleton prediction model (needed for inference) run the following:
```
bash ./bash_scripts/train_skeleton_prediction.sh 
```
Model checkpoints will be saved in `checkpoints/` folder.

### Shape generation
Generative training consists of two independent stages: skeleton diffusion conditioned on shape label and surface diffusion conditioned on skeleton and shape label. 

#### Skeleton diffusion
Model is conditioned on shape label and is trained on ground truth skeletons only. 
To train it, run the following
```
bash ./bash_scripts/train_ldm_skeletons.sh 
```
#### Surface diffusion
Model is conditioned on shape label and skeleton. It is trained in two stages: 1) linear VAE to normalize autoencoder latents to standard gaussian distribution 2) surface diffusion model on encoded latents.  Both stages require encoder from pretrained skeleton autoencoder.

To train latent VAE run the following
```
bash ./bash_scripts/train_latent_vae.sh 
```
To train surface diffusion run the following
```
bash ./bash_scripts/train_ldm_latents.sh 
```

## Preprocessing 
Data preparation take watertight mesh as input and produces shape skeleton graph (we don't use connectivity data in our paper), enveloping implicit function sampling and simplified skeletons for training of skeleton prediction model. It is strongly advised to run skeleton extraction on GPU that supports CUDA (it does not need to be powerful).

To run preprocessing on sample inputs, run the following
```
bash ./bash_scripts/preprocess_data.sh 
``` 
We provide sample outputs of this script in `sample_data/preprocessing/sample_outputs/`. For detailed description of our preprocessing refer to the paper.

## Preprocessed data

You can download preprocessed data using link below:

https://console.cloud.google.com/storage/browser/gem3d_data/data?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&hl=en&project=prime-elf-175923

Folder 'data' contains ShapeNet categories' folders. In each folder you fill find the following:

`skeletons_min_sdf_iter_50` -- processed skeletons

`skelrays_min_sdf_iter_50_1000rays_random` -- sampled enveloping implicit function

`4_pointcloud` -- pointclouds from 3DILG/3DShape2VecSet data and scaling factors (important for evaluation)

`occupancies_compact` -- storage optimized occupancies from 3DILG/3DShape2VecSet data: if you want to train our model, you don't need occupancies there, just want train/val/test splits.

If you have any question about the data, feel free to open the issue or email me (Dmitrii Petrov). 

## Acknowledgements

First and foremost we want to thank [Biao Zhang](https://1zb.github.io/) and his co-authors for coming up with efficient irregular latent grid representation and releasing their data and code for [3DILG](https://github.com/1zb/3DILG) and [3DShape2VecSet](https://github.com/1zb/3DShape2VecSet) papers. Without them, this paper wouldn't be possible. 

We thank Joseph Black, Rik Sengupta and Alexey Goldis for providing useful feedback and input to our project. We acknowledge funding from Adobe Research, the EU H2020 Research Innovation Programme, and the Republic of Cyprus through the Deputy Ministry of Research, Innovation and Digital Policy (GA 739578). 
