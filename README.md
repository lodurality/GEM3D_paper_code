
# GEM3D: Generative Medial Abstractions for 3D Shape Synthesis

### [Project Page](https://lodurality.github.io/GEM3D/) | [Paper (arXiv)](https://arxiv.org/abs/2402.16994)

**Official PyTorch implementation of GEM3D (SIGGRAPH 2024)**

## Quickstart
Download model checkpoints [here](https://drive.google.com/drive/u/3/folders/1vgpzSmFjg61YDlYTIY4kDo28yYC-r02V). Place them in the `submission_checkpoints` folder in the root of the repository. Refer to TODO for detailed checkpoint description.

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

## Training

Since our dataset is fairly large we provide commands to run training on sample data. For full scale training download our data and replace links accordingly. Download instructions and data structure description are provided HERE (TODO). 

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

## Preprocessing (TODO)

## Evaluation (TODO)

## Acknowledgements (TODO)
