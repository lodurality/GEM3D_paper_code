import json
import os
from pathlib import Path
import argparse
import trimesh
import numpy as np
import gc
import pymeshlab as ml

parser = argparse.ArgumentParser(description='Copy meshes data.')
parser.add_argument('--in_path', type=str,
                    help='Path to input folder', required=True)

parser.add_argument('--out_path', type=str,
                    help='Path to output folder', required=True)


parser.add_argument('--num_faces', type=int,
                    default=50000,
                    help='Target number of faces')

parser.add_argument('--chunk_size', type=int, default=100)
parser.add_argument('--chunk_id', type=int, default=0)
parser.add_argument('--folder_to_parse', type=str, default='2_watertight/')
parser.add_argument('--output_folder', type=str, default='watertight_simple/')
parser.add_argument('--output_format', type=str, default='ply')
                    
args = parser.parse_args()

filter_params = {"targetfacenum":args.num_faces,
                 "targetperc" : 0,
                 "qualitythr" : 0.5,
                 "preserveboundary" : False,
                 "boundaryweight" : 1,
                 "preservenormal" : True,
                 "preservetopology" : True,
                 "optimalplacement" : True,
                 "planarquadric" : True,
                 "qualityweight" : False,
                 "autoclean": True,
                 "selected" : False}


folders = sorted(os.listdir(args.in_path))
folders = sorted([item for item in folders if item[:1] == '0'])
all_ids = []
for folder in folders:
    cur_ids = sorted(os.listdir(os.path.join(args.in_path, folder, args.folder_to_parse)))
    cur_ids = [(folder,item) for item in cur_ids]
    all_ids += cur_ids

chunks = [all_ids[i:i + args.chunk_size] for i in range(0, len(all_ids), args.chunk_size)]

cur_chunk = chunks[args.chunk_id]

for item in cur_chunk:
    category, mesh_file = item
    file_name = os.path.join(args.in_path, category, args.folder_to_parse, mesh_file)
    print(f'Simplifying mesh {file_name}...')
    out_folder = os.path.join(args.out_path, category, args.output_folder)
    os.makedirs(out_folder, exist_ok=True)

    try:
        ms = ml.MeshSet()
        ms.load_new_mesh(file_name)
        ms.apply_filter('simplification_quadric_edge_collapse_decimation', **filter_params)
        mesh = ms.mesh(0)
        mesh = trimesh.Trimesh(vertices=mesh.vertex_matrix(), faces=mesh.face_matrix())
        print(f'Simplified mesh is watertight {mesh.is_watertight}')
        out_path = os.path.join(out_folder, mesh_file[:-3] + args.output_format)
        print('Saving simplified mesh to', out_path)
        mesh.export(out_path)
        print("Done. \n=====\n")
    except:
        print('\n\n\n============')
        print(f"Failed to simplify mesh {file_name}")

