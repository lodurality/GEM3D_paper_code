SCRIPT_PATH=/home/$USER/Repos/gem3d_paper_code/

data_path=/home/$USER/Repos/gem3d_paper_code/sample_data/ShapeNet/
num_epoch=501
batch_size=20
lr=0.0001
checkpoint_each=10
suffix=custom
encoder_model_path=/home/$USER/Repos/gem3d_paper_code/submission_checkpoints/skelnet_skelrays_reg_abs_loss_min_sdf_1000_rays_8_heads.pth

python -u $SCRIPT_PATH/train_latent_vae.py --dataset_folder=$data_path --num_epoch=$num_epoch --batch_size=$batch_size --suffix=$suffix --lr=$lr --encoder_model_path=$encoder_model_path --kl_latent_dim=256
