SCRIPT_PATH=/home/$USER/Repos/gem3d_paper_code/preprocessing_scripts
data_path=/home/$USER/Repos/gem3d_paper_code/sample_data/preprocessing/input/
simple_mesh_folder=watertight_simple
num_faces=200000 #num faces for simplification
chunk_size=10 #chunk size for parallelization
chunk_id=0 # chunk id; usually something like $SLURM_ARRAY_TASK_ID
skel_folder=skeletons_min_sdf_iter_50 #skeleton folder
skel_num_iter=50 #number of iterations for skeleton compute
skel_nn=8 #number of nearest neighbors for envelope size estimation; determines average envelope size
skel_downsample_k=1024 #downsampling size for skeletons
reg_sample=40000 #size of regularization sample
num_sphere_points=1000 #number of directions to estimate enveloping implicit function for
sampling_mode=random #mode for sampling directions
skelray_output_folder=skelrays_min_sdf_iter_50_1000rays_random #output folder for final samples
skel_downsample_k=10000 #downsampling size for skeletons for optional step

## Mesh simplification to speedup subsequent steps
echo '=====MESH SIMPLIFICATION====='

python -u $SCRIPT_PATH/simplify_mesh_shapenet.py --in_path=$data_path --out_path=$data_path --folder_to_parse=4_watertight_scaled --chunk_id=$chunk_id --chunk_size=$chunk_size --num_faces=$num_faces --output_format=off --output_folder=$simple_mesh_folder
echo ''

## Generating skeletons using simplified meshes; requires GPU for fast computation
echo '=====SKELETON COMPUTATION====='

python -u $SCRIPT_PATH/generate_skeletons_shapenet.py --data_path=$data_path --out_path=$data_path --folder_to_parse=watertight_simple --chunk_id=$chunk_id --chunk_size=$chunk_size --output_folder=$skel_folder --ignore_starting_zero --num_iter=$skel_num_iter --use_min_sdf_skel

## Generating enveloping implicit function sampling
echo '=====ENVELOPING IMPLICIT FUNCTION COMPUTATION====='


python -u $SCRIPT_PATH/generate_skelrays.py --data_path=$data_path --out_path=$data_path --mesh_folder=watertight_simple --chunk_id=$chunk_id --chunk_size=$chunk_size --output_folder=$skelray_output_folder --ignore_starting_zero --skel_folder=$skel_folder --skel_nn=$skel_nn --skel_downsample_k=$skel_downsample_k --return_reg_points --reg_sample=$reg_sample --num_sphere_points=$num_sphere_points --sampling_mode=$sampling_mode --load_min_sdf --store_directions

## (Optional) Downsample and clean skeletons; needed for training of skeleton prediction model
echo '=====SKELETON DOWNSAMPLING====='


python -u $SCRIPT_PATH/clean_and_downsample_ply_skeletons.py --data_path=$data_path --out_path=$data_path --mesh_folder=watertight_simple --chunk_id=$chunk_id --chunk_size=$chunk_size --output_folder=$skel_folder --ignore_starting_zero --skel_folder=$skel_folder --load_min_sdf --skel_downsample_k=$skel_downsample_k


