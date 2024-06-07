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
parser.add_argument('--skel_folder', type=str, default='skeletons_config_meso_ply_1')
parser.add_argument('--output_folder', type=str, default='skelrays/')
parser.add_argument('--input_format', type=str, default='obj')
parser.add_argument('--num_skeleton_subsamples', type=int, default=1)
parser.add_argument('--skel_downsample_k', type=int, default=576)
parser.add_argument('--vertex_filter', type=int, default=4000)
parser.add_argument('--backend', type=str, default='skimage')
parser.add_argument('--ignore_starting_zero', action='store_true', default=False)
parser.add_argument('--folder_filter', type=str, default=None)
parser.add_argument('--skel_nn', type=int, default=3)
parser.add_argument('--return_reg_points', action='store_true', default=False)
parser.add_argument('--reg_sample', type=int, default=50000)
parser.add_argument('--sampling_mode', type=str, default='icosphere')
parser.add_argument('--num_sphere_points', type=int, default=500)
parser.add_argument('--icosphere_subdivisions', type=int, default=3)
parser.add_argument('--rotate_dirs', action='store_true', default=False)
parser.add_argument('--store_directions', action='store_true', default=False)
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

        print(f'Computing enveloping implicit function for shape {category} {model_id}')
        if args.load_ply:

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

        vals = get_sdfs(skel, gt_mesh)
        skel = skel[vals < 0]
        full_skel = deepcopy(skel)
        skel, _ = fps(skel, K=args.skel_downsample_k)

        res = get_ray_if(skel, gt_mesh, skel_nn=args.skel_nn,
                         sampling_mode=args.sampling_mode,
                         num_sphere_points=args.num_sphere_points,
                         rotate_dirs=args.rotate_dirs,
                         icosphere_subdivisions=args.icosphere_subdivisions)

        patches_nn, ray_dists_nn, directions_orig, skel_nn, mask_nn = res

        out_folder = os.path.join(args.out_path, category, args.output_folder)
        os.makedirs(out_folder, exist_ok=True)
        out_file = os.path.join(out_folder, f'{model_id}_skelrays.npz')

        if not args.store_directions:
            directions_orig = np.array([])

        if args.return_reg_points:
            reg_ray_dists, reg_rays, reg_skel_inds = get_reg_pairs(gt_mesh, skel_nn)
            reg_inds = np.random.randint(0, len(reg_ray_dists), size=args.reg_sample)

            reg_ray_dists, reg_rays, reg_skel_inds = reg_ray_dists[reg_inds],\
                                                     reg_rays[reg_inds],\
                                                     reg_skel_inds[reg_inds]

            np.savez(out_file, ray_dists=ray_dists_nn.astype(np.float16),
                     skel=skel_nn.astype(np.float16),
                     surf_mask=np.packbits(mask_nn),
                     directions=directions_orig.astype(np.float32),
                     reg_ray_dists=reg_ray_dists.astype(np.float16),
                     reg_rays=reg_rays.astype(np.float16),
                     reg_skel_inds=reg_skel_inds.astype(np.int16))
        else:
            np.savez(out_file, ray_dists=ray_dists_nn.astype(np.float16),
                 skel=skel_nn.astype(np.float16),
                 surf_mask=np.packbits(mask_nn),#.astype(bool),
                 directions=directions_orig.astype(np.float16))

        print(f'Done with shape {item}')
    except:
        print(f'Failed to process shape {item}')

