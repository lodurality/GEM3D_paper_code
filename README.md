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

## Preprocessing 
Data preparation take watertight mesh as input and produces shape skeleton graph (we don't use connectivity data in our paper), enveloping implicit function sampling and simplified skeletons for training of skeleton prediction model. It is strongly advised to run skeleton extraction on GPU that supports CUDA (it does not need to be powerful).

To run preprocessing on sample inputs, run the following
```
bash ./bash_scripts/preprocess_data.sh 
``` 
We provide sample outputs of this script in `sample_data/preprocessing/sample_outputs/`. For detailed description of our preprocessing refer to TODO.

## Preprocessed data

You can download preprocessed data using link below:

https://console.cloud.google.com/storage/browser/gem3d_data/data?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&hl=en&project=prime-elf-175923

Folder 'data' contains ShapeNet categories' folders. In each folder you fill find the following:

skeletons_min_sdf_iter_50 -- processed skeletons
skelrays_min_sdf_iter_50_1000rays_random -- sampled enveloping implicit function
4_pointcloud -- pointclouds from 3DILG/3DShape2VecSet data and scaling factors (important for evaluation)
occupancies_compact -- storage optimized occupancies from 3DILG/3DShape2VecSet data: if you want to train our model, you don't need occupancies there, just want train/val/test splits.

If you have any question about the data, feel free to open the issue or email me (Dmitrii Petrov). 

## Acknowledgements

First and foremost we want to thank Biao Zhang and his co-authors for coming up with efficient irregular latent grid representation and releasing their data and code for [3DILG](https://github.com/1zb/3DILG) and [3DShape2VecSet](https://github.com/1zb/3DShape2VecSet) papers. Without them, this paper wouldn't be possible. 

We thank Joseph Black, Rik Sengupta and Alexey Goldis for providing useful feedback and input to our project. We acknowledge funding from Adobe Research, the EU H2020 Research Innovation Programme, and the Republic of Cyprus through the Deputy Ministry of Research, Innovation and Digital Policy (GA 739578). 
