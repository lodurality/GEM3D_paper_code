# adapted from https://github.com/1zb/3DShape2VecSet/blob/master/sample_class_cond.py

import argparse
import math
import numpy as np
import mcubes
import torch
import trimesh
from pathlib import Path
from models.var_ae import KLAutoEncoder, StandarScaler
from models.ldm import EDMPrecondSkel, EDMPrecondSkelCloud
import os
from models.skelnet import SkelAutoencoder
from models.vecset_encoder import VecSetEncoder
from utils.reconstruction import get_mesh_from_latent_combination
from utils.visual import get_o3d_mesh, get_o3d_cloud
import sys
from time import time
import warnings


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # category mapper
    category_ids = {
        '02691156': 0, '02747177': 1, '02773838': 2, '02801938': 3, '02808440': 4, '02818832': 5,
        '02828884': 6, '02843684': 7, '02871439': 8, '02876657': 9, '02880940': 10, '02924116': 11,
        '02933112': 12, '02942699': 13,'02946921': 14, '02954340': 15, '02958343': 16, '02992529': 17,
        '03001627': 18, '03046257': 19, '03085013': 20, '03207941': 21, '03211117': 22, '03261776': 23,
        '03325088': 24, '03337140': 25, '03467517': 26, '03513137': 27, '03593526': 28, '03624134': 29,
        '03636649': 30, '03642806': 31, '03691459': 32, '03710193': 33, '03759954': 34, '03761084': 35,
        '03790512': 36, '03797390': 37, '03928116': 38, '03938244': 39, '03948459': 40, '03991062': 41,
        '04004475': 42, '04074963': 43, '04090263': 44, '04099429': 45, '04225987': 46, '04256520': 47,
        '04330267': 48, '04379243': 49, '04401088': 50, '04460130': 51, '04468005': 52, '04530566': 53,
        '04554684': 54,
    }

    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--ae-pth', type=str, required=True)
    parser.add_argument('--skel_dm_pth', type=str, required=True)
    parser.add_argument('--latent_dm_pth', type=str, required=True)# 'output/uncond_dm/kl_d512_m512_l16_edm/checkpoint-999.pth'
    #parser.add_argument('--latent_encoder_path', type=str, required=True)
    parser.add_argument('--kl_latent_dim', type=int, required=True)
    parser.add_argument('--num_skel_samples', type=int, default=512)
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--num_iters', type=int, default=5)
    parser.add_argument('--skel_seed', type=int, default=0)
    parser.add_argument('--latent_seed', type=int, default=0)
    parser.add_argument('--seed_offset', type=int, default=0)
    parser.add_argument('--resolution', type=int, default=128)
    parser.add_argument('--point_bs', type=int, default=50000)
    parser.add_argument('--skel_nn', type=int, default=1)
    parser.add_argument('--trunc_val', type=float, default=None)
    parser.add_argument('--out_folder', type=str, required=True)
    parser.add_argument('--category', type=str, default=None)
    parser.add_argument('--check_exist', action='store_true', default=False)
    parser.add_argument('--use_spherical_reconstruction', action='store_true', default=False)
    parser.add_argument('--ball_nn', type=int, default=2)
    parser.add_argument('--ball_margin', type=float, default=0.02)
    parser.add_argument('--save_skeleton', action='store_true', default=False)
    parser.add_argument('--use_standard_scaler', action='store_true', default=False)
    parser.add_argument('--load_skeleton', action='store_true', default=False)
    parser.add_argument('--generate_only_skeleton', action='store_true', default=False)
    parser.add_argument('--skel_suffix', type=str, default='')
    parser.add_argument('--surf_suffix', type=str, default='')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_skel_steps', type=int, default=20)
    parser.add_argument('--num_surf_steps', type=int, default=20)


    args = parser.parse_args()
    #print(args)
    out_file_path = args.out_folder + f'/{args.category}-{args.skel_seed}-{args.latent_seed}{args.surf_suffix}.ply'

    if os.path.exists(out_file_path) and args.check_exist:
        print('Mesh exists, skipping generation')
        sys.exit()
    else:
        print('Generating shape', out_file_path)
    device = args.device

    skel_model = EDMPrecondSkelCloud(n_latents=args.num_skel_samples, channels=3, depth=24)

    os.makedirs(args.out_folder, exist_ok=True)

    skel_model.load_state_dict(torch.load(args.skel_dm_pth)['model'])
    skel_model.eval()
    skel_model.to(device)


    category_id = category_ids[args.category]
    skel_model.n_latents = args.num_skel_samples
    # seed offset for evaluation purposes
    seed_offset = args.seed_offset*10000

    # skeleton generation
    # load skeleton option is for precomputed skeletons

    if args.load_skeleton:
        skel_path = args.out_folder + f'/{args.category}_gen_skel_seed_{args.skel_seed}{args.skel_suffix}.npy'
        print('Loading sampled skeletons from ', skel_path)
        sampled_skeleton = np.load(skel_path)
        sampled_skeleton = sampled_skeleton*2
        sampled_skeleton = torch.FloatTensor(sampled_skeleton)
        print('Skeleton shape:', sampled_skeleton.shape)
    else:
        with torch.no_grad():
            sampled_skeleton = skel_model.sample(None, labels=torch.Tensor([category_id]).long().to(device),
                                             batch_seeds=torch.LongTensor([args.skel_seed + seed_offset]).to(device),
                                             num_steps=args.num_skel_steps,
                                             trunc_val=args.trunc_val).float()
            print('Skeleton shape:', sampled_skeleton.shape)

    sampled_skeleton = sampled_skeleton/2
    if args.save_skeleton and not args.load_skeleton:
        skel_path = args.out_folder + f'/{args.category}_gen_skel_seed_{args.skel_seed}{args.skel_suffix}.npy'
        print('Saving sampled skeletons to', skel_path)
        np.save(skel_path, sampled_skeleton.detach().cpu().numpy())
    del skel_model
    torch.cuda.empty_cache()

    if args.generate_only_skeleton:
        sys.exit('Was instructed to generate only skeleton.')

    if args.use_standard_scaler:
        latent_encoder_object = StandarScaler
    else:
        latent_encoder_object = KLAutoEncoder


    latent_model = EDMPrecondSkel(n_latents=args.num_skel_samples, channels=256, depth=24,
                           latent_encoder_object=latent_encoder_object,
                           latent_dim=args.kl_latent_dim)

    latent_model.load_state_dict(torch.load(args.latent_dm_pth)['model'])
    latent_model.eval()
    latent_model.to(device)
    sampled_skeleton = sampled_skeleton.to(device)

    # surface generation
    with torch.no_grad():
        sampled_latents = latent_model.sample(skeletons=sampled_skeleton,
                                     labels=torch.Tensor([category_id]).long().to(device),
                                     batch_seeds=torch.LongTensor([args.latent_seed + seed_offset]).to(device),
                                     num_steps=args.num_surf_steps).float()
        decoded_latents = latent_model.latent_encoder.decode(sampled_latents)

    #saving some memory
    del latent_model
    torch.cuda.empty_cache()
    ###  reconstruction
    ae = SkelAutoencoder(use_skel_model=False, surface_num=2048,
                         encoder_object=VecSetEncoder,
                         skel_model_object=None,
                         num_encoder_heads=8,
                         num_disps=1)

    ae.eval()
    ae.load_state_dict(torch.load(args.ae_pth, map_location='cpu'))

    ae.to(device)

    o3dmesh, recon_mesh, IF = get_mesh_from_latent_combination(decoded_latents, sampled_skeleton,
                                                               ae,
                                                               args.resolution,
                                                               level=0.0, skel_nn=args.skel_nn,
                                                               padding=8,
                                                               max_dimensions=0.5 * np.array([1, 1, 1]),
                                                               min_dimensions=0.5 * np.array([-1, -1, -1]),
                                                               bs=args.point_bs,
                                                               use_spherical_reconstruction=args.use_spherical_reconstruction,
                                                               ball_nn=args.ball_nn,
                                                               ball_margin=args.ball_margin)

    recon_mesh.vertices *= 2
    mesh_path = args.out_folder + f'/{args.category}-{args.skel_seed}-{args.latent_seed}{args.surf_suffix}.ply'
    print('Saving final mesh to', mesh_path)
    recon_mesh.export(mesh_path)
    print('Done')
