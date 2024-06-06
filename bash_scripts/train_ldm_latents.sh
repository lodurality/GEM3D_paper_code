SCRIPT_PATH=/home/$USER/Repos/gem3d_paper_code/ #path to script
data_path=/home/$USER/Repos/gem3d_paper_code/sample_data/ShapeNet/ #path to data

model=skel_kl_d512_m2048_l256_d24_edm
skel_ae_path=/home/$USER/Repos/gem3d_paper_code/submission_checkpoints/skelnet_skelrays_reg_abs_loss_min_sdf_1000_rays_8_heads.pth
output_dir=checkpoints/dm/latent_ldm/
batch_size=1 #1 for code release; 20 for training in the paper
epochs=1000
blr=0.0001
num_workers=6
latent_encoder_path=/home/$USER/Repos/gem3d_paper_code/submission_checkpoints/kl_ae_skeletons_min_sdf_iter_50_dim_256_2080ti.pth
kl_latent_dim=256
checkpoint_each=30
warmup_epochs=1
point_cloud_size=4096


# for training on cluster replace "python -u" with "#torchrun --nproc_per_node=8"

python -u $SCRIPT_PATH/train_ldm_latents.py --accum_iter 2 --model $model  --ae-pth $skel_ae_path --output_dir $output_dir --log_dir $output_dir --num_workers $num_workers --point_cloud_size 2048 --batch_size $batch_size --epochs $epochs  --data_path $data_path --skeleton_conditioning --blr $blr --latent_encoder_path $latent_encoder_path --kl_latent_dim=$kl_latent_dim --occupancies_base_folder=occupancies --checkpoint_each $checkpoint_each --warmup_epochs=$warmup_epochs --point_cloud_size=$point_cloud_size --use_fps


