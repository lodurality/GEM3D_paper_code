import os
import numpy as np
import trimesh
import sys
import torch

from utils.data import Surf2SkeletonShapeNet, Surf2SkeletonShapeNetMemory, collate_reprs_finetuning
from utils.p2pnet_utils import compute_chamfer
from utils.spatial import get_topk_nns_dilated, get_flat_queries_and_centers
from models.skelnet import SkelAutoencoder
from models.vecset_encoder import VecSetEncoder, P2PNetVecSetEncoder
from models.point_transformer import P2PNetPointTransformer
from utils.p2pnet_utils import compute_chamfer_and_density
import argparse
import json
from time import time
from models.var_ae import KLAutoEncoder
from functools import partial

def make_train_step_volume(data, model, encoder_model, optimizer, args, train=True):
    if train:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()
    gt_skel, surface = data[1], data[0]
    # print(gt_skel.shape)

    gt_skel = gt_skel.to(device)
    surface = surface.to(device)
    queries = data[2].to(device)
    labels = data[3].to(device)

    if encoder_model.use_skel_model:
        cloud_disp = model.skel_model(surface)
        B, M, D = surface.shape
        cloud_disp = cloud_disp.reshape(B, M, model.skel_model.num_disps, D)
        cloud2skel = surface[:,:,None,:] + cloud_disp
        cloud2skel = cloud2skel.reshape(B, -1, D)
    else:
        cloud2skel = gt_skel.clone()

    with torch.no_grad():
        latents, centers, (row, col) = encoder_model.encoder.forward(surface, cloud2skel)

    #flat_latents = latents.reshape(-1,latents.shape[-1])
    kl, latents_decode = model(latents)
    recon_loss = torch.linalg.norm(latents.reshape(-1, latents.shape[-1]) - latents_decode.reshape(-1,latents.shape[-1]),
                                   axis=-1).mean(axis=0)  # .mean(axis=0)
    loss = recon_loss + args.kl_weight * kl.mean()

    n = len(surface)

    if train:
        loss.backward()
        optimizer.step()

    return n, loss.detach().mean().item(), recon_loss.item(),\
           kl.detach().mean().item(), 0


def run_one_epoch(loader, model, encoder_model, optimizer, args, train=True):
    cum_results = [0, 0, 0, 0, 0]

    denom = 0
    for i, data in enumerate(loader):
        train_step_results = make_train_step_volume(data, model, encoder_model, optimizer, args, train=train)
        cur_n = train_step_results[0]
        for i in range(len(train_step_results[1:])):
            cum_results[i] += cur_n * train_step_results[i + 1]
        denom += cur_n
    cum_results = tuple([item / denom for item in cum_results])

    return cum_results


parser = argparse.ArgumentParser(description='P2P-NET')
parser.add_argument('--dataset_folder', type=str, default='/home/dmpetrov/data/ShapeNetDILG/',
                    help='file of the names of the point clouds')
parser.add_argument('--encoder_model_path', type=str, required=True)
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
parser.add_argument('--checkpoint_path', type=str, default='checkpoints/vae_ae/')
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
parser.add_argument('--skel_folder_basename', type=str, default='skeletons_min_sdf_iter_50')
parser.add_argument('--kl_latent_dim', type=int, default=256)
parser.add_argument('--num_encoder_heads', type=int, default=8)
parser.add_argument('--kl_weight', type=float, default=0.01)

args = parser.parse_args()

os.makedirs(args.checkpoint_path, exist_ok=True)

model_prefix = f'{args.checkpoint_path}/kl_ae_{args.skel_folder_basename}_dim_{args.kl_latent_dim}_{args.suffix}'


with open(model_prefix + '.json', 'w+') as f:
    json.dump(vars(args), f, indent=4)

if args.categories is not None:
    categories = args.categories.split()
    print('Categories to load', categories)
else:
    categories = None
    print('Loading all categories')

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
                                            load_npz_skeletons=True)

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

skel_model_object = None
encoder_model = SkelAutoencoder(use_skel_model=False, surface_num=2048,
                       encoder_object=VecSetEncoder,
                        use_aggregator=args.use_aggregator,
                        agg_skel_nn=args.agg_skel_nn,
                        use_quantizer=False,
                        skel_model_object=skel_model_object,
                        num_encoder_heads=args.num_encoder_heads,
                        num_disps=1).to(device)


encoder_model.load_state_dict(torch.load(args.encoder_model_path))
model = KLAutoEncoder(dim=256, latent_dim=args.kl_latent_dim).to(device)

if args.pretrained_path is not None:
    print(f'Loading model from {args.pretrained_path}')
    model.load_state_dict(torch.load(args.pretrained_path), strict=False)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

best_loss = 1e6
for epoch_idx in range(args.num_epoch):
    # data = collate_reprs_p2p([train_dataset[ind]])
    start = time()
    train_results = run_one_epoch(train_loader, model, encoder_model, optimizer, args, train=True)

    print("Train epoch {} loss is {:.4f} (recon {:.4f}, kl {:.4f}). Elapsed time {:.3f} s.".format(
        *((epoch_idx,) + train_results[:3] + (time() - start, ))))

    val_results = run_one_epoch(val_loader, model, encoder_model, optimizer, args, train=False)

    print("Val epoch {} loss is {:.4f} (recon {:.4f}, kl {:.5f})".format(
        *((epoch_idx,) + val_results[:3])))

    if val_results[0] < best_loss:
        print(f'Got new best val loss {round(val_results[0], 5)} (previous was {best_loss})')
        best_loss = round(val_results[0], 5)
        torch.save(model.state_dict(), model_prefix + '.pth')

    if epoch_idx % args.checkpoint_each == 0:
        torch.save(model.state_dict(), model_prefix + f'_{epoch_idx}.pth')

    # print('==\n')
