import os
import sys
import numpy as np
import trimesh
import argparse
import signal


sys.path.append('/home/dmpetrov/Repos/shape_generation/')
sys.path.append('/home/dmpetrov_umass_edu/Repos/top_shape_generation')

from utils.skeleton import get_mcfskel_file, get_full_min_sdf_skeleton
from utils.run_utils import time_limit

parser = argparse.ArgumentParser(description='Copy meshes data.')
parser.add_argument('--data_path', type=str,
                    help='Path to data', required=True)
parser.add_argument('--out_path', type=str,
                    help='Path to output folder', required=True)
parser.add_argument('--chunk_size', type=int, default=100)
parser.add_argument('--resolution', type=int, default=300)
parser.add_argument('--timeout_limit', type=int, default=120)
parser.add_argument('--chunk_id', type=int, default=0)
parser.add_argument('--folder_to_parse', type=str, default='')
parser.add_argument('--output_folder', type=str, default='skeletons/')
parser.add_argument('--input_format', type=str, default='obj')
parser.add_argument('--num_skeleton_subsamples', type=int, default=1)
parser.add_argument('--tet_graph_points', type=int, default=512)
parser.add_argument('--vertex_filter', type=int, default=4000)
parser.add_argument('--backend', type=str, default='skimage')
parser.add_argument('--mcfskel_path', type=str, default='')
parser.add_argument('--mcfskel_file', type=str, default='mcfskel')
parser.add_argument('--config_path', type=str, default='config.json')
parser.add_argument('--ignore_starting_zero', action='store_true', default=False)
parser.add_argument('--folder_filter', type=str, default=None)
parser.add_argument('--save_meso_ply', action='store_true', default=False)
parser.add_argument('--use_mcfskel', action='store_true', default=False)
parser.add_argument('--use_min_sdf_skel', action='store_true', default=False)
parser.add_argument('--num_iter', type=int, default=50)
parser.add_argument('--lsds_mult', type=float, default=0.6)



args = parser.parse_args()
#print(args)
assert args.use_mcfskel != args.use_min_sdf_skel

folders = sorted(os.listdir(args.data_path))
if not args.ignore_starting_zero:
    folders = sorted([item for item in folders if item[:1] == '0'])

if args.folder_filter is not None:
    print('Filtering folders')
    folders = [item for item in folders if args.folder_filter in item]
    print('After filtering', folders)

all_ids = []
for folder in folders:
    cur_ids = sorted(os.listdir(os.path.join(args.data_path, folder, args.folder_to_parse)))
    cur_ids = [(folder,item) for item in cur_ids]
    all_ids += cur_ids

chunks = [all_ids[i:i + args.chunk_size] for i in range(0, len(all_ids), args.chunk_size)]
cur_chunk = chunks[args.chunk_id]

mcfskel_path = os.path.join(args.mcfskel_path, args.mcfskel_file)
config_path = os.path.join(args.mcfskel_path, args.config_path)

#print('mcfskel_path', mcfskel_path)
#print('config path', config_path)


for item in cur_chunk:
    try:
        category, mesh_file = item
        cur_file = os.path.join(args.data_path, category, args.folder_to_parse, mesh_file)
        print('Computing skeleton for mesh ', cur_file)

        if args.use_mcfskel:
            print('THIS OPTION IS DEPRECATED IN CODE RELEASE: PROVIDED AS REFERENCE')
            file_prefix = mesh_file[:-4] #+ f'_{args.config_path.replace(".", "_")}'
            out_folder = os.path.join(args.out_path, category, args.output_folder)
            os.makedirs(out_folder, exist_ok=True)
            skel_out = os.path.join(out_folder, file_prefix + '_skel.txt')
            corr_out = os.path.join(out_folder, file_prefix + '_corr.txt')

            if args.save_meso_ply:
                meso_out = os.path.join(out_folder, file_prefix + '_meso.ply')
            else:
                meso_out = None

            output, error = get_mcfskel_file(cur_file, mcfskel_path, config_path,
                                             skel_out, corr_out, mesoskel_out=meso_out)
            if output is not None:
                print(output.decode())
            if error is not None:
                print(error.decode())
            print(f'Done with shape {item}')
        elif args.use_min_sdf_skel:
            orig_mesh = trimesh.load(cur_file)
            mesh = orig_mesh.subdivide_to_size(max_edge=0.05, max_iter=50)
            file_prefix = mesh_file[:-4]  # + f'_{args.config_path.replace(".", "_")}'
            out_folder = os.path.join(args.out_path, category, args.output_folder)
            os.makedirs(out_folder, exist_ok=True)
            skel_out = os.path.join(out_folder, file_prefix + '_skel_graph.npz')
            skel_points, edges = get_full_min_sdf_skeleton(mesh, num_iter=args.num_iter,
                                                           lsds_mult=args.lsds_mult)
            pts_save = skel_points.numpy().astype(np.float32)
            edges_save = edges.astype(np.int32)

            np.savez(skel_out, vertices=pts_save, edges=edges_save)


        else:
            raise ValueError('No skeleton type specified.')

    except:
        print(f'Failed to process shape {item}')

