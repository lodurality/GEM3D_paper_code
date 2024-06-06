SCRIPT_PATH=/home/$USER/Repos/gem3d_paper_code/
data_path=/home/$USER/Repos/gem3d_paper_code/sample_data/ShapeNet/
num_epoch=1001
batch_size=14
lr=0.0001
checkpoint_each=100
suffix=p2p_min_sdf_vecset
agg_skel_nn=1
num_queries_sample=100
skel_folder_basename=skeletons_min_sdf_iter_50
num_disps=4
num_skel_samples=8192

python -u $SCRIPT_PATH/train_skeleton_prediction.py --dataset_folder=$data_path --num_epoch=$num_epoch --batch_size=$batch_size --suffix=$suffix --lr=$lr --agg_skel_nn=$agg_skel_nn --num_queries_sample=$num_queries_sample --use_skel_model  --skel_model_type=vecset --skel_folder_basename=$skel_folder_basename --num_disps=$num_disps --num_skel_samples=$num_skel_samples 
