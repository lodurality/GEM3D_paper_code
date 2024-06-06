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
parser.add_argument('--data_path', type=str, default='/home/dmpetrov/data/ShapeNetDILG/',
                    help='file of the names of the point clouds')
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--skelmodel_path', type=str, default=None)
parser.add_argument('--agg_model_path', type=str, default=None)
parser.add_argument('--point_bs', type=int, default=50000)
parser.add_argument('--out_path', type=str, required=True)
parser.add_argument('--stats_path', type=str, default='stats/train_val_test_split')
parser.add_argument('--ignore_starting_zero', action='store_true', default=False)
parser.add_argument('--folder_filter', type=str, default=None)
parser.add_argument('--folder_to_parse', type=str, default='')
parser.add_argument('--split', type=str, default='test')
parser.add_argument('--categories', type=str, default=None)
parser.add_argument('--chunk_size', type=int, default=2)
parser.add_argument('--resolution', type=int, default=256)
parser.add_argument('--timeout_limit', type=int, default=120)
parser.add_argument('--chunk_id', type=int, default=0)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--init_surface_sample', type=int, default=20000)
parser.add_argument('--num_skel_nn', type=int, default=64)
parser.add_argument('--joint_patches', action='store_true', default=False)
parser.add_argument('--use_meshes', action='store_true', default=False)
parser.add_argument('--filter_meshes', action='store_true', default=False)
parser.add_argument('--use_quantizer', action='store_true', default=False)
parser.add_argument('--use_skel_model', action='store_true', default=False)
parser.add_argument('--skel_nn', type=int, default=1)
parser.add_argument('--skelray_folder', type=str, default='skelrays_512_4')
parser.add_argument('--num_skel_samples', type=int, default=512)
parser.add_argument('--use_aggregator', action='store_true', default=False)
parser.add_argument('--agg_skel_nn', type=int, default=1)
parser.add_argument('--skeleton_folder_basename', type=str, default='skeletons_config_meso_ply_1')
parser.add_argument('--use_projections', action='store_true', default=False)
parser.add_argument('--use_line_projections', action='store_true', default=False)
parser.add_argument('--use_projected_skeleton', action='store_true', default=False)
parser.add_argument('--use_weighted_neighbors', action='store_true', default=False)
parser.add_argument('--use_projected_latents', action='store_true', default=False)
parser.add_argument('--use_smoothing', action='store_true', default=False)
parser.add_argument('--kernel_width', type=float, default=1.0)
parser.add_argument('--num_encoder_heads', type=int, default=1)
parser.add_argument('--num_disps', type=int, default=1)
parser.add_argument('--skel_model_type', type=str, default='point_transformer')
parser.add_argument('--disp_aggregation', type=str, default=None)
parser.add_argument('--check_exist', action='store_true', default=False)
parser.add_argument('--use_con_model', action='store_true', default=False)
parser.add_argument('--con_model_path', type=str, default=False)
parser.add_argument('--con_model_thres', type=float, default=0.5)
parser.add_argument('--use_spherical_reconstruction', action='store_true', default=False)
parser.add_argument('--ball_nn', type=int, default=3)
parser.add_argument('--ball_margin', type=float, default=0.01)
parser.add_argument('--shape_scale', type=float, default=1.0)
parser.add_argument('--folder_suffix', type=str, default='')
parser.add_argument('--dataset', type=str, default='shapenet')

args = parser.parse_args()

device = args.device
point_folder = os.path.join(args.data_path, '')
if args.categories is None:
    categories = os.listdir(point_folder)
    categories = [c for c in categories if
                  os.path.isdir(os.path.join(point_folder, c)) and c.startswith('0')]
else:
    categories = args.categories.split(',')

categories.sort()

all_ids = []

error_message_data = '--dataset option must be either "shapenet" or "thingi10k"'
error_message_thingi = 'Split for Thingi10K dataset must be "all"'
assert args.dataset == 'shapenet' or args.dataset == "thingi10k", error_message_data
if args.dataset == 'thingi10k':
    assert args.split == 'all', error_message_thingi

split_folder = 'occupancies' if args.dataset == 'shapenet' else 'raw_meshes'
for c_idx, c in enumerate(categories):
    subpath = os.path.join(point_folder, c, split_folder)
    # print(subpath)
    assert os.path.isdir(subpath)

    split_file = os.path.join(subpath, args.split + '.lst')
    with open(split_file, 'r') as f:
        models_c = f.read().split('\n')
        models_c = [item for item in models_c if item != '']

    all_ids += [
        {'category': c, 'model': m.replace('.npz', '')}
        for m in models_c
    ]

data_chunks = [all_ids[i:i + args.chunk_size] for i in range(0, len(all_ids), args.chunk_size)]
cur_data_chunk = data_chunks[args.chunk_id]

print('cur_chunk')
print(cur_data_chunk)

if args.dataset == 'shapenet':

    eval_dataset = Surf2SkeletonShapeNetMemory(dataset_folder=args.data_path, split='test',
                                                occupancies_folder='occupancies',
                                                load_skeletons=True,
                                                categories=categories,
                                                skeleton_folder_basename=args.skeleton_folder_basename,
                                                subsample=None,
                                                surface_sampling=False,
                                                load_occupancies=False,
                                                num_samples=100000,
                                                pc_size=100000,
                                                return_dijkstras=False,
                                                load_normals=False,
                                                load_correspondences=False,
                                                is_compact=True,
                                                return_dict_getitem=True,
                                                skelray_folder=args.skelray_folder,
                                                load_skelrays=False,
                                                near_points_share=0.95,
                                                load_npz_skeletons=True,
                                                load_simple_occupancies=False,
                                                simple_occ_folder='occ_simple',
                                                ids_to_load=cur_data_chunk)
    print(f'Finished loading ShapeNet data, overall length is {len(eval_dataset)}')

else:
    print('Thingi10K dataset size is ', len(all_ids))
    eval_dataset = None

use_skel_model = args.use_skel_model

if use_skel_model:
    skel_model_object=P2PNetVecSetEncoder
else:
    skel_model_object=None

model = SkelAutoencoder(use_skel_model=use_skel_model, use_skel_correspondences=False,
                        surface_num=2048,
                       num_skel=512,num_skel_nn=32,
                       encoder_object=VecSetEncoder,
                        skel_model_object=skel_model_object,
                        agg_skel_nn=args.agg_skel_nn,
                        num_encoder_heads=args.num_encoder_heads,
                        num_disps=args.num_disps)#.to(device)


model_skel = SkelAutoencoder(use_skel_model=use_skel_model, use_skel_correspondences=False,
                        surface_num=2048,
                       num_skel=512,num_skel_nn=32,
                       encoder_object=VecSetEncoder,
                        skel_model_object=skel_model_object,
                        agg_skel_nn=args.agg_skel_nn,
                        num_encoder_heads=1,
                             num_disps=args.num_disps)#.to(device)

model_name = args.model_path.split('/')[-1][:-4]

if use_skel_model:
    skelmodel_name = args.skelmodel_path.split('/')[-1][:-4]
    model_name += '__' + skelmodel_name

if args.use_spherical_reconstruction:
    print('Using spherical recon')
    model_name += '__' + 'ball_recon' + f'__{str(args.ball_nn).replace(".","")}'
else:
    model_name += '__' + 'naive_recon' + f'__{str(args.skel_nn).replace(".", "")}'

model_name += '__res_' + str(args.resolution) + '__' + str(args.num_skel_samples)
model_name += args.folder_suffix
model_name = args.dataset + '_' + model_name
out_folder = args.out_path + '/' + model_name + '/' + 'recon/'

if not os.path.exists(out_folder):
    os.makedirs(out_folder, exist_ok=True)

model.load_state_dict(torch.load(args.model_path), strict=False)

if use_skel_model:
    print('Loading skel model')
    model_skel.load_state_dict(torch.load(args.skelmodel_path), strict=False)
    model.skel_model = model_skel.skel_model


model.to(device)
model.eval()

collate_reprs = partial(collate_reprs_simple, num_surface_sample=100000,
                        num_skel_samples=args.num_skel_samples,
                        )

os.makedirs(args.out_path, exist_ok=True)

eval_models = eval_dataset.clean_models if args.dataset == 'shapenet' else cur_data_chunk

for check_ind, model_dict in enumerate(eval_models):
    #try:
    category_id, model_id = model_dict['category'], model_dict['model']
    if os.path.exists(out_folder + f'{category_id}_{model_id}.ply') and args.check_exist:
        print('Mesh exists, skipping reconstruction')
        continue

    with torch.no_grad():
        model.eval()
        if args.dataset == 'shapenet':
            data = collate_reprs([eval_dataset[check_ind]])
            gt_skel, surface = data[1], data[0]
            gt_skel = gt_skel.type(torch.float32).to(device)
        else:
            gt_skel = None
            thingi_shape = trimesh.load(f'{args.data_path}/{category_id}/raw_meshes/{model_id}', force='mesh')
            thingi_shape.vertices = thingi_shape.vertices - thingi_shape.bounding_box.vertices.mean(axis=0)
            thingi_shape.vertices /= thingi_shape.bounding_box.extents.max()
            surface = torch.FloatTensor((thingi_shape.sample(100000))).unsqueeze(0)
            model_id = model_id.split('.')[0]

        surface = surface.type(torch.float32).to(device)
        surface_encode = simple_fps(surface[0], N=2048)
        surface_encode = surface_encode.unsqueeze(0)

        if model.use_skel_model:
            print('Using skel model')
            cloud_disp = model.skel_model(surface_encode)
            cloud2skel = surface_encode[:, :, None, :] + cloud_disp.reshape(len(cloud_disp), cloud_disp.shape[1], -1, 3)
            if args.disp_aggregation is not None and args.disp_aggregation == 'mean':
                print('Doing mean aggregation')
                cloud2skel = torch.mean(cloud2skel, axis=2)
            if args.disp_aggregation is not None and args.disp_aggregation == 'median':
                print('Doing skel median aggregation')
                cloud2skel = torch.median(cloud2skel, axis=2)[0]

            cloud2skel = cloud2skel.reshape(len(cloud2skel), -1, 3)
            orig_cloud2skel = cloud2skel.clone()
        else:
            cloud2skel = gt_skel.clone()

        cloud2skel = simple_fps(cloud2skel[0], N=args.num_skel_samples)
        cloud2skel = cloud2skel.unsqueeze(0)

        latents, centers, (row, col) = model.encode(surface_encode, cloud2skel)

    o3dmesh, recon_mesh, IF = get_mesh_from_latent_combination(latents, centers,
                                                               model,
                                                               args.resolution,
                                                               level=0.0, skel_nn=args.skel_nn,
                                                               padding=5,
                                                               max_dimensions=0.5*args.shape_scale*np.array([1, 1, 1]),
                                                               min_dimensions=0.5*args.shape_scale*np.array([-1, -1, -1]),
                                                               shape_scale=args.shape_scale,
                                                               bs=args.point_bs,
                                                               use_spherical_reconstruction=args.use_spherical_reconstruction,
                                                               ball_nn=args.ball_nn,
                                                               ball_margin=args.ball_margin)

    recon_mesh.export(out_folder + f'{category_id}_{model_id}.ply')
    print(f'Saved model to {out_folder}/{category_id}_{model_id}.ply')

    if model.use_skel_model:
        skel_path = out_folder + f'{category_id}_{model_id}_skeleton.npz'
        np.savez(skel_path, skel=cloud2skel[0].detach().cpu().numpy())


    #except:
    #    print(f'Failed to reconstruct mesh', model_id)
    #draw(get_o3d_mesh(tst), get_o3d_cloud(ref_points[0].detach().cpu().numpy(), color=[0, 0, 1]))

