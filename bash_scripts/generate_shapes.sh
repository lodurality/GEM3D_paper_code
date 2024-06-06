SCRIPT_PATH=/home/$USER/Repos/gem3d_paper_code/
ae_pth=/home/$USER/Repos/gem3d_paper_code/submission_checkpoints/skelnet_skelrays_reg_abs_loss_min_sdf_1000_rays_8_heads_1200.pth 
skel_dm_pth=/home/$USER/Repos/gem3d_paper_code/submission_checkpoints/checkpoint_skel_ldm_1024.pth
latent_dm_pth=/home/$USER/Repos/gem3d_paper_code/submission_checkpoints/checkpoint_l256_fin.pth
out_folder=./custom_outputs/generation/
category=$1
num_skel_steps=100
num_surf_steps=50
num_skel_samples=2048
skel_seed=$2
latent_seed=$3
point_bs=30000

python $SCRIPT_PATH/generate_shapes.py  --ae-pth $ae_pth --skel_dm_pth $skel_dm_pth --kl_latent_dim 256 --out_folder $out_folder --category=$1 --num_skel_steps $num_skel_steps --num_surf_steps $num_surf_steps --num_skel_samples $num_skel_samples --latent_dm_pth $latent_dm_pth --skel_seed $skel_seed --latent_seed $latent_seed --point_bs $point_bs --ball_nn=3 --save_skeleton --use_spherical_reconstruction
