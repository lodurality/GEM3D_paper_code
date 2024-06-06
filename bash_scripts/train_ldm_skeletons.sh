SCRIPT_PATH=/home/$USER/Repos/gem3d_paper_code/ #path to script

model=skel_ldm
skel_ae_path=/home/$USER/Repos/gem3d_paper_code/submission_checkpoints/skelnet_skelrays_reg_abs_loss_min_sdf_1000_rays_8_heads.pth
num_skel_samples=1024
output_dir=checkpoints/dm/skeleton_ldm/
batch_size=1 #1 for sample data; 8 for training
epochs=1000
data_path=/home/$USER/Repos/gem3d_paper_code/sample_data/ShapeNet/ #path to data
blr=0.0002
num_workers=4
checkpoint_each=30
warmup_epochs=1

python -u $SCRIPT_PATH/train_ldm_skeletons.py --accum_iter 2 --model $model  --output_dir $output_dir --log_dir $output_dir --num_workers $num_workers --batch_size $batch_size --epochs $epochs  --data_path $data_path --skeleton_conditioning --blr $blr --occupancies_base_folder=occupancies --checkpoint_each $checkpoint_each --warmup_epochs=$warmup_epochs --use_dilg_scale --num_skel_samples=$num_skel_samples

# uncomment to train on cluster
#torchrun --nproc_per_node=8 $SCRIPT_PATH/train_ldm_skeletons.py --accum_iter 2 --model $model  --ae kl_d512_m512_l8  --ae-pth $skel_ae_path --output_dir $output_dir --log_dir $output_dir --num_workers $num_workers --batch_size $batch_size --epochs $epochs  --data_path $data_path --skeleton_conditioning --blr $blr --occupancies_base_folder=occupancies --checkpoint_each $checkpoint_each --warmup_epochs=$warmup_epochs --use_dilg_scale --num_skel_samples=$num_skel_samples

