import os
import numpy as np
import trimesh
import sys
import torch

from utils.data import Surf2SkeletonShapeNet, Surf2SkeletonShapeNetMemory, collate_reprs_finetuning, AxisScaling
from utils.p2pnet_utils import compute_chamfer
from utils.spatial import get_topk_nns_dilated, get_flat_queries_and_centers, get_geodesic_skel_patches
from models.skelnet import SkelAutoencoder
from models.vecset_encoder import VecSetEncoder, P2PNetVecSetEncoder
from models.point_transformer import P2PNetPointTransformer
from utils.p2pnet_utils import compute_chamfer_and_density
import argparse
import json
from time import time


def make_train_step_volume(data, model, optimizer, args, train=True):
    if train:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()
    gt_skel, surface = data[1], data[0]
    gt_skel = gt_skel.to(device)
    surface = surface.to(device)

    if model.use_skel_model:
        cloud_disp = model.skel_model(surface)
        B, M, D = surface.shape
        cloud_disp = cloud_disp.reshape(B, M, model.skel_model.num_disps, D)
        cloud2skel = surface[:,:,None,:] + cloud_disp
        cloud2skel = cloud2skel.reshape(B, -1, D)
    else:
        cloud2skel = gt_skel.clone()

    if model.use_skel_model:
        skel_chamfer_loss, skel_density_loss = compute_chamfer_and_density(cloud2skel, gt_skel)

    skel_density_loss = args.density_loss_weight*skel_density_loss
    loss = skel_chamfer_loss.mean() + skel_density_loss.mean()
    n = len(surface)

    if train:
        loss.backward()
        optimizer.step()

    return n, loss.detach().mean().item(), skel_chamfer_loss.detach().mean().item(),\
           skel_density_loss.detach().mean().item(), 0


def run_one_epoch(loader, model, optimizer, args, train=True):
    cum_results = [0, 0, 0, 0, 0]

    denom = 0
    for i, data in enumerate(loader):
        train_step_results = make_train_step_volume(data, model, optimizer, args, train=train)
        cur_n = train_step_results[0]
        for i in range(len(train_step_results[1:])):
            cum_results[i] += cur_n * train_step_results[i + 1]
        denom += cur_n
    cum_results = tuple([item / denom for item in cum_results])

    return cum_results


parser = argparse.ArgumentParser(description='P2P-NET')
parser.add_argument('--dataset_folder', type=str, default='/home/dmpetrov/data/ShapeNetDILG/',
                    help='file of the names of the point clouds')
parser.add_argument('--data_subsample', type=int, default=None)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_epoch', type=int, default=201)
parser.add_argument('--num_surface_queries', type=int, default=512)
parser.add_argument('--exp_gamma', type=float, default=0.999)
parser.add_argument('--scale_loss_weight', type=float, default=1.0)
parser.add_argument('--skel_loss_weight', type=float, default=1.0)
parser.add_argument('--use_skel_model', action='store_true', default=False)
parser.add_argument('--exp_name', type=str, default='')
parser.add_argument('--surface_samples', type=int, default=1024)
parser.add_argument('--fin_samples', type=int, default=256)
parser.add_argument('--dist_thres', type=float, default=0.02)
parser.add_argument('--skel_k', type=int, default=5)
parser.add_argument('--num_surface_sample', type=int, default=2048)
parser.add_argument('--num_skel_samples', type=int, default=2048)
parser.add_argument('--occ_samples_collate', type=int, default=512)
parser.add_argument('--checkpoint_each', type=int, default=1000)
parser.add_argument('--suffix', type=str, default='test')
parser.add_argument('--hinge_weight', type=float, default=1.0)
parser.add_argument('--scale_weight', type=float, default=1.0)
parser.add_argument('--chamfer_weight', type=float, default=1.0)
parser.add_argument('--checkpoint_path', type=str, default='checkpoints/p2p_meso_attn/')
parser.add_argument('--categories', type=str, default=None)
parser.add_argument('--skelray_folder', type=str, default='skelrays_512_4')
parser.add_argument('--num_queries_sample', type=int, default=4000)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--pretrained_path', type=str, default=None)
parser.add_argument('--agg_skel_nn', type=int, default=3)
parser.add_argument('--sdf_loss_mult', type=float, default=1e4)
parser.add_argument('--smooth_loss_mult', type=float, default=0)
parser.add_argument('--use_quantizer', action='store_true', default=False)
parser.add_argument('--use_aggregator', action='store_true', default=False)
parser.add_argument('--train_skel_nn', type=int, default=1)
parser.add_argument('--dir_consistency', action='store_true', default=True)
parser.add_argument('--cosine_thres', type=float, default=0.86)
parser.add_argument('--density_loss_weight', type=float, default=0.2)
parser.add_argument('--skel_model_type', type=str, default='attn')
parser.add_argument('--skel_folder_basename', type=str, default='skeletons_config_meso_ply_1')
parser.add_argument('--num_disps', type=int, default=1)
parser.add_argument('--augment_data', action='store_true', default=False)


args = parser.parse_args()

os.makedirs(args.checkpoint_path, exist_ok=True)

model_prefix = f'{args.checkpoint_path}/p2p_{args.skel_model_type}_{args.skel_folder_basename}_{args.suffix}'

with open(model_prefix + '.json', 'w+') as f:
    json.dump(vars(args), f, indent=4)

if args.categories is not None:
    categories = args.categories.split()
    print('Categories to load', categories)
else:
    categories = None
    print('Loading all categories')

if args.augment_data:
    transform=AxisScaling()
else:
    transform=None

train_dataset = Surf2SkeletonShapeNetMemory(dataset_folder=args.dataset_folder, split='train',
                                            occupancies_folder='occupancies',
                                            load_skeletons=True,
                                            categories=categories,
                                            skeleton_folder_basename=args.skel_folder_basename,
                                            subsample=args.data_subsample,
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
                                            transform=transform)

val_dataset = Surf2SkeletonShapeNetMemory(dataset_folder=args.dataset_folder, split='val',
                                          occupancies_folder='occupancies',
                                          load_skeletons=True,
                                          skeleton_folder_basename=args.skel_folder_basename,
                                          categories=categories,
                                          subsample=args.data_subsample,
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
                                          load_npz_skeletons=True)
init_skel_size = train_dataset[0]['skeleton'][0].shape[0]
print('Init skel size', init_skel_size)
collate_reprs = partial(collate_reprs_finetuning, num_surface_sample=args.num_surface_sample,
                        num_skel_samples=args.num_skel_samples,
                        num_queries_sample=args.num_queries_sample, return_surface_queries=True,
                        init_skel_size=init_skel_size)


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                           collate_fn=collate_reprs, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                         collate_fn=collate_reprs, shuffle=False)

device = args.device

if args.skel_model_type == 'attn':
    skel_p2p = P2PNetVecSetEncoder
elif args.skel_model_type == 'trans':
    print('Warning: not used in the final version of the paper.')
    skel_p2p = P2PNetPointTransformer
elif args.skel_model_type == 'vecset':
    skel_p2p = P2PNetVecSetEncoder

else:
    raise ValueError('Wrong skel_model_type: must be attn/trans')

model = SkelAutoencoder(use_skel_model=True, surface_num=2048,
                       encoder_object=VecSetEncoder,
                        use_aggregator=args.use_aggregator,
                        skel_model_object=skel_p2p,
                        agg_skel_nn=args.agg_skel_nn,
                        use_quantizer=args.use_quantizer,
                        num_disps=args.num_disps).to(device)

if args.pretrained_path is not None:
    print(f'Loading model from {args.pretrained_path}')
    model.load_state_dict(torch.load(args.pretrained_path), strict=False)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

best_loss = 1e6
for epoch_idx in range(args.num_epoch):
    start = time()
    train_results = run_one_epoch(train_loader, model, optimizer, args, train=True)

    print("Train epoch {} loss is {:.4f} (chamfer {:.4f}, density {:.4f}). Elapsed time {:.3f} s.".format(
        *((epoch_idx,) + train_results[:3] + (time() - start, ))))

    val_results = run_one_epoch(val_loader, model, optimizer, args, train=False)

    print("Val epoch {} loss is {:.4f} (chamfer {:.4f}, density {:.5f})".format(
        *((epoch_idx,) + val_results[:3])))

    if val_results[0] < best_loss:
        print(f'Got new best val loss {round(val_results[0], 5)} (previous was {best_loss})')
        best_loss = round(val_results[0], 5)
        torch.save(model.state_dict(), model_prefix + '.pth')

    if epoch_idx % args.checkpoint_each == 0:
        torch.save(model.state_dict(), model_prefix + f'_{epoch_idx}.pth')
