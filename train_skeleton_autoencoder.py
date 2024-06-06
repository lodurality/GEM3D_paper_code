import os
import numpy as np
import trimesh
import sys
import torch

from utils.data import Surf2SkeletonShapeNetMemory, collate_reprs_skelrays, AxisScalingSkelray
from models.skelnet import SkelAutoencoder
from models.vecset_encoder import VecSetEncoder
import argparse
import json
from functools import partial
#from utils.visual import *


def make_train_step_volume(data, model, optimizer, args, train=True):
    if train:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()
    gt_skel, surface = data[1], data[0]

    gt_skel = gt_skel.to(device)
    surface = surface.to(device)
    ray_queries = data[2].to(device)
    ray_scales = data[3].to(device)
    skel_ids = data[4].to(device)

    rays_reg = data[-2].to(device)
    reg_ray_dists = data[-3].to(device)
    reg_skel_inds = data[-1].to(device)

    rays_reg_flat = rays_reg.reshape(rays_reg.shape[0], -1, rays_reg.shape[-1])
    reg_scales_flat = reg_ray_dists.reshape(reg_ray_dists.shape[0], -1, reg_ray_dists.shape[-1])
    reg_skel_inds_flat = reg_skel_inds.reshape(reg_skel_inds.shape[0], -1)

    ray_queries = torch.cat([ray_queries, rays_reg_flat], axis=1)
    ray_scales = torch.cat([ray_scales, reg_scales_flat], axis=1)
    skel_ids = torch.cat([skel_ids, reg_skel_inds_flat], axis=1)

    if model.use_skel_model:
        cloud_disp = model.skel_model(surface)
        cloud2skel = surface + cloud_disp
    else:
        cloud2skel = gt_skel.clone()

    latents, centers, (row, col) = model.encoder.forward(surface, cloud2skel)

    all_preds = model.decoder.forward(latents, centers, ray_queries, skel_ids)
    all_preds = torch.sigmoid(all_preds)
    gt_scales = ray_scales[:,:,0]

    pred_reg_scales = all_preds[:, -reg_scales_flat.shape[1]:]
    pred_reg_scales = pred_reg_scales.reshape(reg_ray_dists.shape)
    reg_skel_pts = gt_skel[torch.arange(gt_skel.size(0)).unsqueeze(1), reg_skel_inds_flat]
    reg_skel_pts = reg_skel_pts.reshape(rays_reg.shape)

    surface_reg = reg_skel_pts + pred_reg_scales*rays_reg

    reg_loss = torch.square(surface_reg[:,:, 0, :] - surface_reg[:,:, 1, :]).sum(dim=-1).mean(axis=-1)

    if args.abs_loss:
        scale_loss = (torch.abs(all_preds - gt_scales)).mean(
            axis=-1)
    else:
        scale_loss = (torch.abs(torch.log(all_preds + 1) - torch.log(gt_scales + 1))).mean(
            axis=-1)
    reg_loss = args.reg_weight*reg_loss
    reg_loss = torch.clamp(reg_loss, max=args.reg_clamp)
    loss = (scale_loss + reg_loss).mean()

    n = len(surface)

    if train:
        loss.backward()
        optimizer.step()

    return n, loss.detach().mean().item(), scale_loss.mean().item(), reg_loss.mean().item()


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
parser.add_argument('--num_skel_samples', type=int, default=512)
parser.add_argument('--occ_samples_collate', type=int, default=512)
parser.add_argument('--checkpoint_each', type=int, default=1000)
parser.add_argument('--suffix', type=str, default='test')
parser.add_argument('--hinge_weight', type=float, default=1.0)
parser.add_argument('--scale_weight', type=float, default=1.0)
parser.add_argument('--chamfer_weight', type=float, default=1.0)
parser.add_argument('--checkpoint_path', type=str, default='checkpoints/skelnet_skelrays_reg/')
parser.add_argument('--pretrained_path', type=str, default=None)
parser.add_argument('--categories', type=str, default=None)
parser.add_argument('--skelray_folder', type=str, default='skelrays_512_4_reg')
parser.add_argument('--num_ray_queries', type=int, default=1000)
parser.add_argument('--num_reg_sample', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--reg_weight', type=float, default=200)
parser.add_argument('--reg_clamp', type=float, default=0.005)
parser.add_argument('--unpack_bits', action='store_true', default=False)
parser.add_argument('--abs_loss', action='store_true', default=False)
parser.add_argument('--num_encoder_heads', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--non_surface_weight', type=float, default=0.1)
parser.add_argument('--augment_data', action='store_true', default=False)
parser.add_argument('--sample_proportional_to_disp', action='store_true', default=False)
parser.add_argument('--jitter_skel_supervision', action='store_true', default=False)
parser.add_argument('--skel_jitter_scale', type=float, default=0.005)

args = parser.parse_args()

os.makedirs(args.checkpoint_path, exist_ok=True)
with open(f'{args.checkpoint_path}/skelnet_skelray_reg_{args.suffix}.json', 'w+') as f:
    json.dump(vars(args), f, indent=4)

if args.categories is not None:
    categories = args.categories.split()
    print('Categories to load', categories)
else:
    categories = None
    print('Loading all categories')

train_dataset = Surf2SkeletonShapeNetMemory(dataset_folder=args.dataset_folder, split='train',
                                            occupancies_folder='occupancies',
                                            load_skeletons=False,
                                            categories=categories,
                                            skeleton_folder_basename='skeletons_config_json',
                                            subsample=args.data_subsample,
                                            surface_sampling=False,
                                            load_occupancies=False,
                                            num_samples=10000,
                                            pc_size=40000,
                                            return_dijkstras=False,
                                            load_normals=False,
                                            load_correspondences=False,
                                            is_compact=True,
                                            return_dict_getitem=True,
                                            skelray_folder=args.skelray_folder,
                                            load_skelrays=True,
                                            load_reg_skelrays=True)

val_dataset = Surf2SkeletonShapeNetMemory(dataset_folder=args.dataset_folder, split='val',
                                          occupancies_folder='occupancies',
                                          load_skeletons=False,
                                          skeleton_folder_basename='skeletons_config_json',
                                          categories=categories,
                                          subsample=args.data_subsample,
                                          surface_sampling=False,
                                          load_occupancies=False,
                                          num_samples=10000,
                                          pc_size=40000,
                                          return_dijkstras=False,
                                          load_normals=False,
                                          load_correspondences=False,
                                          is_compact=True,
                                          return_dict_getitem=True,
                                          skelray_folder=args.skelray_folder,
                                          load_skelrays=True,
                                          load_reg_skelrays=True)

if args.augment_data:
    transform = AxisScalingSkelray()
else:
    transform = None

collate_reprs_train = partial(collate_reprs_skelrays, num_surface_sample=args.num_surface_sample,
                        num_ray_queries=args.num_ray_queries, subsample_skels=True,
                        num_skel_samples=args.num_skel_samples, return_reg_data=True,
                        num_reg_sample=args.num_reg_sample,
                        unpack_bits=args.unpack_bits,
                        non_surface_weight=args.non_surface_weight,
                        transform=transform,
                        sample_proportional_to_disp=args.sample_proportional_to_disp,
                        jitter_skel_supervision=args.jitter_skel_supervision,
                        jitter_scale=args.skel_jitter_scale)

collate_reprs_val = partial(collate_reprs_skelrays, num_surface_sample=args.num_surface_sample,
                        num_ray_queries=args.num_ray_queries, subsample_skels=True,
                        num_skel_samples=args.num_skel_samples, return_reg_data=True,
                        num_reg_sample=args.num_reg_sample,
                        unpack_bits=args.unpack_bits,
                        non_surface_weight=args.non_surface_weight,
                        transform=None,
                        sample_proportional_to_disp=args.sample_proportional_to_disp)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                           collate_fn=collate_reprs_train, shuffle=True,
                                           num_workers=args.num_workers)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                         collate_fn=collate_reprs_val, shuffle=False,
                                         num_workers=args.num_workers)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SkelAutoencoder(use_skel_model=False, surface_num=2048,
                       encoder_object=VecSetEncoder,
                       num_encoder_heads=args.num_encoder_heads)
#print(model)
if args.pretrained_path is not None:
    model.load_state_dict(torch.load(args.pretrained_path))


if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    print('Running model on several GPUs')
    model = MyDataParallel(model).to(device)
else:
    model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


best_loss = 1e6

for epoch_idx in range(args.num_epoch):
    # data = collate_reprs_p2p([train_dataset[ind]])

    train_results = run_one_epoch(train_loader, model, optimizer, args, train=True)

    print("Train epoch {} loss is {:.4f} (scale {:.4f}, reg {:.4f})".format(
        *((epoch_idx,) + train_results[:3])))

    val_results = run_one_epoch(val_loader, model, optimizer, args, train=False)

    print("Val epoch {} loss is {:.4f} (scale {:.4f}, reg {:.4f} )".format(
        *((epoch_idx,) + val_results[:3])))

    if val_results[0] < best_loss:
        print(f'Got new best val loss {round(val_results[0], 5)} (previous was {best_loss})')
        best_loss = round(val_results[0], 5)
        torch.save(model.state_dict(), f'{args.checkpoint_path}/skelnet_skelrays_reg_{args.suffix}.pth')

    if epoch_idx % args.checkpoint_each == 0:
        torch.save(model.state_dict(), f'{args.checkpoint_path}/skelnet_skelrays_reg_{args.suffix}_{epoch_idx}.pth')

    # print('==\n')
