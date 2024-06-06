SCRIPT_PATH=/home/$USER/Repos/gem3d_paper_code/ #path to script
data_path=/home/$USER/Repos/gem3d_paper_code/sample_data/Thingi10K/ #path to data
model_path=/home/$USER/Repos/gem3d_paper_code/submission_checkpoints/skelnet_skelrays_reg_abs_loss_min_sdf_1000_rays_8_heads.pth
skelmodel_path=/home/$USER/Repos/gem3d_paper_code/submission_checkpoints/p2p_vecset_skeletons_min_sdf_iter_50_p2p_min_sdf_vecset_from_ckpt.pth 
out_path=./custom_outputs/reconstruction/
chunk_size=10 #chunk size for parallelization
chunk_id=0 #chunk id for parallelization
skeleton_folder_basename=skeletons_min_sdf_iter_50 #gt skeleton folder
point_bs=20000
resolution=128 #128 for the code release; 256 for evaluation in the paper

python -u $SCRIPT_PATH/reconstruct_dataset.py --model_path=$model_path --skelmodel_path=$skelmodel_path --out_path=$out_path --chunk_size=10 --skeleton_folder_basename=$skeleton_folder_basename --use_skel_model --data_path=$data_path --point_bs=$point_bs --skel_nn=1 --num_skel_samples=2048 --num_disps=4 --num_encoder_heads=8 --skel_model_type=vecset --disp_aggregation=median --use_spherical_reconstruction --ball_nn=2 --ball_margin=0.02 --shape_scale=1.02 --resolution=$resolution --dataset=thingi10k --split=all  --chunk_id=$chunk_id
