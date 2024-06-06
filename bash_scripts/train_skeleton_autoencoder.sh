SCRIPT_PATH=/home/$USER/Repos/gem3d_paper_code/ #path to script
data_path=/home/$USER/Repos/gem3d_paper_code/sample_data/ShapeNet/ #path to data
num_epoch=2001 #final model was trained for 1000 epochs
batch_size=20 #might change depending on your GPU; this is batch size for RTX 2080ti
lr=0.00007
checkpoint_each=100
num_ray_queries=10000
skelray_folder=skelrays_min_sdf_iter_50_1000rays_random
suffix=${skelray_folder}_reg_abs_loss_8_heads

python -u $SCRIPT_PATH/train_skeleton_autoencoder.py --dataset_folder=$data_path --num_epoch=$num_epoch --batch_size=$batch_size --suffix=$suffix --lr=$lr --num_ray_queries=$num_ray_queries --skelray_folder=$skelray_folder --unpack_bits --abs_loss --num_encoder_heads=8 
