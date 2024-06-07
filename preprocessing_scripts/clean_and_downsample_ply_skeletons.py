import os
import sys
import numpy as np
import trimesh
import argparse
from copy import deepcopy


sys.path.append('/home/dmpetrov/Repos/shape_generation/')
sys.path.append('/home/dmpetrov_umass_edu/Repos/top_shape_generation')

from utils.spatial import get_ray_if, get_sdfs, fps, get_reg_pairs

parser = argparse.ArgumentParser(description='Copy meshes data.')
parser.add_argument('--data_path', type=str,
                    help='Path to data', required=True)
parser.add_argument('--out_path', type=str,
                    help='Path to output folder', required=True)
parser.add_argument('--chunk_size', type=int, default=100)
parser.add_argument('--resolution', type=int, default=300)
parser.add_argument('--timeout_limit', type=int, default=120)
parser.add_argument('--chunk_id', type=int, default=0)
parser.add_argument('--mesh_folder', type=str, default='watertight_simple')
parser.add_argument('--skel_folder', type=str, default='skeletons_config_meso_ply_3')
parser.add_argument('--output_folder', type=str, default='skelrays/')
parser.add_argument('--input_format', type=str, default='obj')
parser.add_argument('--num_skeleton_subsamples', type=int, default=1)
parser.add_argument('--skel_downsample_k', type=int, default=4096)
parser.add_argument('--vertex_filter', type=int, default=4000)
parser.add_argument('--backend', type=str, default='skimage')
parser.add_argument('--ignore_starting_zero', action='store_true', default=False)
parser.add_argument('--folder_filter', type=str, default=None)
parser.add_argument('--skel_nn', type=int, default=3)
parser.add_argument('--return_reg_points', action='store_true')
parser.add_argument('--reg_sample', type=int, default=50000)
parser.add_argument('--load_ply', action='store_true', default=False)
parser.add_argument('--load_min_sdf', action='store_true', default=False)

args = parser.parse_args()


folders = sorted(os.listdir(args.data_path))
if not args.ignore_starting_zero:
    folders = sorted([item for item in folders if item[:1] == '0'])

if args.folder_filter is not None:
    print('Filtering folders')
    folders = [item for item in folders if args.folder_filter in item]
    print('After filtering', folders)

all_ids = []
for folder in folders:
    cur_ids = sorted(os.listdir(os.path.join(args.data_path, folder, args.mesh_folder)))
    cur_ids = [(folder,item) for item in cur_ids]
    all_ids += cur_ids

chunks = [all_ids[i:i + args.chunk_size] for i in range(0, len(all_ids), args.chunk_size)]
cur_chunk = chunks[args.chunk_id]

for item in cur_chunk:
    try:
        category, mesh_file = item
        model_id = mesh_file[:-4]
        mesh_file = os.path.join(args.data_path, category, args.mesh_folder, mesh_file)
        if args.load_ply:
            print('DEPRECATED OPTION FOR CODE RELEASE; PROVIDED AS REFERENCE')
            skel_ply_file = os.path.join(args.data_path, category, args.skel_folder, f'{model_id}_meso.ply')
            skel_ply = trimesh.load(skel_ply_file, force='mesh')
            skel = skel_ply.vertices
        elif args.load_min_sdf:
            skel_file = os.path.join(args.data_path, category, args.skel_folder, f'{model_id}_skel_graph.npz')
            new_skel = np.load(skel_file)
            skel, edges = new_skel['vertices'], new_skel['edges']
        else:
            raise ValueError('Wrong loading argument.')

        gt_mesh = trimesh.load(mesh_file)

        print('Skel shape before sdf filtering', skel.shape)
        vals = get_sdfs(skel, gt_mesh)
        skel = skel[vals < 0]
        print('Skel shape after sdf filtering', skel.shape)
        full_skel = deepcopy(skel)

        if len(skel) > args.skel_downsample_k:
            skel, _ = fps(skel, K=args.skel_downsample_k)

        print('Downsampled skel shape', skel.shape)

        out_folder = os.path.join(args.out_path, category, args.skel_folder)
        #print(out_folder)
        os.makedirs(out_folder, exist_ok=True)
        out_file = os.path.join(out_folder, f'{model_id}_clean_skel.npz')
        np.savez(out_file, skel=skel)
        print(f'Saved downsampled skeleton to {out_file}')
        print(f'Done.')
    except:
        print(f'Failed to process shape {item}')

