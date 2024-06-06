import argparse
import json
import os
import sys

sys.path.append('/home/dmpetrov/Repos/shape_generation/')
sys.path.append('/home/dmpetrov_umass_edu/Repos/top_shape_generation')

import torch
from models.skelnet import *
from utils.spatial import *
from models.point_transformer import P2PNetPointTransformer
from utils.data import Surf2SkeletonShapeNetMemory, collate_reprs_simple
from models.vecset_encoder import VecSetEncoder, P2PNetVecSetEncoder
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from scipy.spatial import KDTree
from utils.reconstruction import get_mesh_from_latent_combination
import gc
import networkx as nx


parser = argparse.ArgumentParser(description='Shape overfitting params')
parser.add_argument('--input_mesh_path', type=str, required=True)
parser.add_argument('--output_folder', type=str, required=True)
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--skelmodel_path', type=str, default=None)
parser.add_argument('--point_bs', type=int, default=50000)
parser.add_argument('--skel_nn', type=int, default=1)
parser.add_argument('--num_skel_samples', type=int, default=512)
parser.add_argument('--use_spherical_reconstruction', action='store_true', default=False)
parser.add_argument('--ball_nn', type=int, default=3)
parser.add_argument('--ball_margin', type=float, default=0.02)
parser.add_argument('--shape_scale', type=float, default=1.02)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--resolution', type=int, default=256)
parser.add_argument('--num_surface_samples', type=int, default=2048)
parser.add_argument('--vis_outputs', action='store_true', default=False)


args = parser.parse_args()

device = args.device

use_skel_model = True
skel_model_object=P2PNetVecSetEncoder

model = SkelAutoencoder(use_skel_model=use_skel_model, use_skel_correspondences=False,
                        surface_num=args.num_surface_samples,
                       num_skel=512,num_skel_nn=32,
                       encoder_object=VecSetEncoder,
                        skel_model_object=skel_model_object,
                        agg_skel_nn=1,
                        num_encoder_heads=8,
                        num_disps=4)#.to(device)


model_skel = SkelAutoencoder(use_skel_model=use_skel_model, use_skel_correspondences=False,
                        surface_num=args.num_surface_samples,
                       num_skel=512,num_skel_nn=32,
                       encoder_object=VecSetEncoder,
                        skel_model_object=skel_model_object,
                        num_encoder_heads=1,
                             num_disps=4)#.to(device)


model.load_state_dict(torch.load(args.model_path), strict=False)
model_skel.load_state_dict(torch.load(args.skelmodel_path), strict=False)
model.skel_model = model_skel.skel_model
model.to(device)
model.eval()

input_shape = trimesh.load(args.input_mesh_path, force_mesh=True)
input_shape.vertices = input_shape.vertices - input_shape.bounding_box.vertices.mean(axis=0)
input_shape.vertices /= input_shape.bounding_box.extents.max()

surface = torch.FloatTensor((input_shape.sample(100000))).unsqueeze(0)
surface = surface.type(torch.float32).to(device)
surface_encode = simple_fps(surface[0], N=args.num_surface_samples)
surface_encode = surface_encode.unsqueeze(0).to(device)

with torch.no_grad():
    cloud_disp = model.skel_model(surface_encode)
    cloud2skel = surface_encode[:, :, None, :] + cloud_disp.reshape(len(cloud_disp), cloud_disp.shape[1], -1, 3)
    cloud2skel = torch.median(cloud2skel, axis=2)[0]
    cloud2skel = cloud2skel.reshape(len(cloud2skel), -1, 3)
    cloud2skel = simple_fps(cloud2skel[0], N=args.num_skel_samples)
    cloud2skel = cloud2skel.unsqueeze(0)

    latents, centers, (row, col) = model.encode(surface_encode, cloud2skel)
    o3dmesh, recon_mesh, IF = get_mesh_from_latent_combination(latents, centers,
                                                               model,
                                                               args.resolution,
                                                               device=args.device,
                                                               level=0.0, skel_nn=1,
                                                               padding=5,
                                                               max_dimensions=0.5*args.shape_scale*np.array([1, 1, 1]),
                                                               min_dimensions=0.5*args.shape_scale*np.array([-1, -1, -1]),
                                                               shape_scale=args.shape_scale,
                                                               bs=args.point_bs,
                                                               use_spherical_reconstruction=True,
                                                               ball_nn=args.ball_nn,
                                                               ball_margin=args.ball_margin)

os.makedirs(args.output_folder, exist_ok=True)
mesh_path = args.output_folder + '/recon_mesh.ply'
recon_mesh.export(mesh_path)
print(f'Saved reconstructed model to {mesh_path}')
scaled_mesh_path = args.output_folder + '/input_mesh_scaled.ply'
input_shape.export(scaled_mesh_path)
print(f'Saved scaled input mesh {mesh_path}')

if model.use_skel_model:
    skel_path = args.output_folder + '/recon_skeleton.npz'
    np.savez(skel_path, skel=cloud2skel[0].detach().cpu().numpy())
    print(f'Saved inferred skeleton to {skel_path}')

if args.vis_outputs:
    shift=1
    from utils.visual import get_o3d_cloud, get_o3d_mesh, draw

    input_o3d = get_o3d_mesh(input_shape, color=[1, 0.706, 0])
    cloud_vis = surface_encode[0].detach().cpu().numpy() + np.array([shift,0,0])
    skel_vis = cloud2skel[0].detach().cpu().numpy() + np.array([shift*2,0,0])
    surf_cloud_o3d = get_o3d_cloud(cloud_vis)
    skel_o3d = get_o3d_cloud(skel_vis)
    recon_mesh.vertices += np.array([shift*3,0,0])
    recon_o3d = get_o3d_mesh(recon_mesh, color=[1,0.706,0])
    draw(input_o3d, surf_cloud_o3d, skel_o3d, recon_o3d)

