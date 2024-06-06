
import os
import glob
import random

import yaml 

import torch
from torch.utils import data

import numpy as np

from PIL import Image

import h5py
from torch_cluster import fps as fps_cluster

def fps_from_cloud(cloud, N=256):

    if len(cloud.shape) == 2:
        cloud = cloud.unsqueeze(0)

    B1, N1, D1 = cloud.shape
    cloud_flat = cloud.view(B1 * N1, D1)
    pos_cloud = cloud_flat

    batch1 = torch.arange(B1).to(cloud.device)
    batch1 = torch.repeat_interleave(batch1, N1)
    idx = fps_cluster(pos_cloud, batch1, ratio=N / N1)  # 0.0625
    #idx = idx[:N]
    #print(idx.shape)
    idx = idx.reshape(len(cloud), N)

    sampled_cloud = pos_cloud[idx]
    batch2 = torch.arange(len(idx)).to(cloud.device)
    batch2 = torch.repeat_interleave(batch2, idx.shape[1]).reshape(len(idx), -1)
    #print(idx.shape, batch2.shape)

    return sampled_cloud, idx - N1 * batch2

category_ids = {
    '02691156': 0,
    '02747177': 1,
    '02773838': 2,
    '02801938': 3,
    '02808440': 4,
    '02818832': 5,
    '02828884': 6,
    '02843684': 7,
    '02871439': 8,
    '02876657': 9, 
    '02880940': 10,
    '02924116': 11,
    '02933112': 12,
    '02942699': 13,
    '02946921': 14,
    '02954340': 15,
    '02958343': 16,
    '02992529': 17,
    '03001627': 18,
    '03046257': 19,
    '03085013': 20,
    '03207941': 21,
    '03211117': 22,
    '03261776': 23,
    '03325088': 24,
    '03337140': 25,
    '03467517': 26,
    '03513137': 27,
    '03593526': 28,
    '03624134': 29,
    '03636649': 30,
    '03642806': 31,
    '03691459': 32,
    '03710193': 33,
    '03759954': 34,
    '03761084': 35,
    '03790512': 36,
    '03797390': 37,
    '03928116': 38,
    '03938244': 39,
    '03948459': 40,
    '03991062': 41,
    '04004475': 42,
    '04074963': 43,
    '04090263': 44,
    '04099429': 45,
    '04225987': 46,
    '04256520': 47,
    '04330267': 48,
    '04379243': 49,
    '04401088': 50,
    '04460130': 51,
    '04468005': 52,
    '04530566': 53,
    '04554684': 54,
}

class ShapeNet(data.Dataset):
    def __init__(self, dataset_folder, split, categories=['03001627'], transform=None,
                 sampling=True, num_samples=4096, return_surface=True, surface_sampling=True,
                 pc_size=2048, replica=16, return_scale=False):
        
        self.pc_size = pc_size

        self.transform = transform
        self.num_samples = num_samples
        self.sampling = sampling
        self.split = split

        self.dataset_folder = dataset_folder
        self.return_surface = return_surface
        self.surface_sampling = surface_sampling

        self.dataset_folder = dataset_folder
        self.return_scale = return_scale

        #self.point_folder = os.path.join(self.dataset_folder, 'ShapeNetV2_point')
        #self.mesh_folder = os.path.join(self.dataset_folder, 'ShapeNetV2_watertight')

        self.point_folder = os.path.join(self.dataset_folder, '')
        self.mesh_folder = os.path.join(self.dataset_folder, '')

        if categories is None:
            categories = os.listdir(self.point_folder)
            categories = [c for c in categories if os.path.isdir(os.path.join(self.point_folder, c)) and c.startswith('0')]
        categories.sort()
        print('Categories')
        print(categories)

        self.models = []
        for c_idx, c in enumerate(categories):
            subpath = os.path.join(self.point_folder, c)
            #print(subpath)
            assert os.path.isdir(subpath)

            split_file = os.path.join(subpath, 'occupancies', split + '.lst')
            with open(split_file, 'r') as f:
                models_c = f.read().split('\n')
            
            self.models += [
                {'category': c, 'model': m.replace('.npz', '')}
                for m in models_c
            ]

        self.replica = replica

    def __getitem__(self, idx):
        idx = idx % len(self.models)

        category = self.models[idx]['category']
        model = self.models[idx]['model']


        point_path = os.path.join(self.point_folder, category, 'occupancies', model+'.npz')
        try:
            with np.load(point_path) as data:
                vol_points = data['vol_points']
                vol_label = data['vol_label']
                near_points = data['near_points']
                near_label = data['near_label']
        except Exception as e:
            print(e)
            print(point_path)

        with open(point_path.replace('.npz', '.npy'), 'rb') as f:
            scale = np.load(f).item()

        if self.return_surface:
            pc_path = os.path.join(self.mesh_folder, category, '4_pointcloud', model+'.npz')
            with np.load(pc_path) as data:
                surface = data['points'].astype(np.float32)
                surface = surface * scale
            if self.surface_sampling:
                ind = np.random.default_rng().choice(surface.shape[0], self.pc_size, replace=False)
                surface = surface[ind]
            surface = torch.from_numpy(surface)

        if self.sampling:
            ind = np.random.default_rng().choice(vol_points.shape[0], self.num_samples, replace=False)
            vol_points = vol_points[ind]
            vol_label = vol_label[ind]

            ind = np.random.default_rng().choice(near_points.shape[0], self.num_samples, replace=False)
            near_points = near_points[ind]
            near_label = near_label[ind]

        
        vol_points = torch.from_numpy(vol_points)
        vol_label = torch.from_numpy(vol_label).float()


        if self.split == 'train':
            near_points = torch.from_numpy(near_points)
            near_label = torch.from_numpy(near_label).float()

            points = torch.cat([vol_points, near_points], dim=0)
            labels = torch.cat([vol_label, near_label], dim=0)
        else:
            points = vol_points
            labels = vol_label

        #if self.transform:
        #    surface, points = self.transform(surface, points)

        ##print(self.return_surface, self.return_scale)
        if self.return_surface and self.return_scale:
            return points, labels, surface, category_ids[category], scale
        elif self.return_surface and not self.return_scale:
            return points, labels, surface, category_ids[category]
        else:
            return points, labels, category_ids[category]

    def __len__(self):
        if self.split != 'train':
            return len(self.models)
        else:
            return len(self.models) * self.replica


class ShapeNetSkel(data.Dataset):
    def __init__(self, dataset_folder, split, categories=None, transform=None,
                 sampling=True, num_samples=4096,
                 return_surface=True, surface_sampling=True, pc_size=2048, replica=16,
                 return_skeleton=True,
                 skeleton_folder_basename='skeletons_min_sdf_iter_50',
                 data_subsample=None,
                 use_dilg_scale=True,
                 occupancies_base_folder='occupancies',
                 num_skel_samples=512,
                 use_fps=False):

        self.pc_size = pc_size
        self.use_dilg_scale = use_dilg_scale
        self.occupancies_base_folder = occupancies_base_folder
        self.transform = transform
        self.num_samples = num_samples
        self.sampling = sampling
        self.split = split
        self.num_skel_samples = num_skel_samples
        self.use_fps = use_fps
        self.dataset_folder = dataset_folder
        self.return_surface = return_surface
        self.return_skeleton = return_skeleton
        self.skeleton_folder_basename = skeleton_folder_basename
        self.surface_sampling = surface_sampling

        self.dataset_folder = dataset_folder

        # self.point_folder = os.path.join(self.dataset_folder, 'ShapeNetV2_point')
        # self.mesh_folder = os.path.join(self.dataset_folder, 'ShapeNetV2_watertight')

        self.point_folder = os.path.join(self.dataset_folder, '')
        self.mesh_folder = os.path.join(self.dataset_folder, '')

        if categories is None:
            categories = os.listdir(self.point_folder)
            categories = [c for c in categories if
                          os.path.isdir(os.path.join(self.point_folder, c)) and c.startswith('0')]
        categories.sort()
        print('Categories')
        print(categories)

        self.models = []
        for c_idx, c in enumerate(categories):
            subpath = os.path.join(self.point_folder, c)
            ##print(subpath)
            assert os.path.isdir(subpath)

            split_file = os.path.join(subpath, self.occupancies_base_folder, split + '.lst')
            with open(split_file, 'r') as f:
                models_c = f.read().split('\n')

            self.models += [
                {'category': c, 'model': m.replace('.npz', '')}
                for m in models_c
            ]

        self.replica = replica

        if data_subsample is not None:
            print('Data subsample', data_subsample)
            self.models = self.models[:data_subsample]

        final_models = []
        for item in self.models:
            category = item['category']
            model = item['model']
            pc_path = os.path.join(self.mesh_folder, category, '4_pointcloud', model + '.npz')
            skeleton_path = os.path.join(self.point_folder, category, self.skeleton_folder_basename,
                                         model + '_clean_skel.npz')

            if os.path.exists(pc_path) and os.path.exists(skeleton_path):
                final_models += [item]

        print('Share of good data', len(final_models)/len(self.models))
        self.models = final_models
        print('Data size')
        print(len(self.models))
        #print(self.models)

    def __getitem__(self, idx):
        idx = idx % len(self.models)

        category = self.models[idx]['category']
        model = self.models[idx]['model']


        point_path = os.path.join(self.point_folder, category, self.occupancies_base_folder, model + '.npz')
        '''
        try:
            with np.load(point_path) as data:
                vol_points = data['vol_points']
                vol_label = data['vol_label']
                near_points = data['near_points']
                near_label = data['near_label']
        except Exception as e:
            print(e)
            print(point_path)
        '''
        if self.use_dilg_scale:
            with open(point_path.replace('.npz', '.npy'), 'rb') as f:
                scale = np.load(f).item()
        else:
            scale = 1


        if self.return_surface:
            pc_path = os.path.join(self.mesh_folder, category, '4_pointcloud', model + '.npz')

            with np.load(pc_path) as data:
                full_surface = data['points'].astype(np.float32)
                #print(full_surface.shape)
                full_surface = full_surface * scale
            if self.surface_sampling:
                ind = np.random.default_rng().choice(full_surface.shape[0], self.pc_size, replace=False)
                surface = full_surface[ind]
                surface = torch.from_numpy(surface)

        if self.return_skeleton:

            skeleton_path = os.path.join(self.point_folder, category, self.skeleton_folder_basename,
                                         model + '_clean_skel.npz')
            # print(skeleton_path)
            full_skel = np.load(skeleton_path)['skel']*scale
            #print(skel.max(), surface.max())
            skel = torch.from_numpy(full_skel).float()
            if self.use_fps:
                skel, _ = fps_from_cloud(skel, N=self.num_skel_samples)
                skel = skel[0]
            else:
                skel_ind = np.random.default_rng().choice(skel.shape[0], self.num_skel_samples, replace=True)
                skel = skel[skel_ind]

        '''
        if self.sampling:
            ind = np.random.default_rng().choice(vol_points.shape[0], self.num_samples, replace=False)
            vol_points = vol_points[ind]
            vol_label = vol_label[ind]

            ind = np.random.default_rng().choice(near_points.shape[0], self.num_samples, replace=False)
            near_points = near_points[ind]
            near_label = near_label[ind]

        vol_points = torch.from_numpy(vol_points)
        vol_label = torch.from_numpy(vol_label).float()

        if self.split == 'train':
            near_points = torch.from_numpy(near_points)
            near_label = torch.from_numpy(near_label).float()

            points = torch.cat([vol_points, near_points], dim=0)
            labels = torch.cat([vol_label, near_label], dim=0)
        else:
            points = vol_points
            labels = vol_label

        if self.transform:
            surface, points = self.transform(surface, points)
        '''

        points, labels = torch.zeros(1), torch.zeros(1)

        if self.return_surface:
            return points, labels, surface, category_ids[category], skel
        else:
            return points, labels, category_ids[category], skel

    def __len__(self):
        if self.split != 'train':
            return len(self.models)
        else:
            return len(self.models) * self.replica


class ShapeNetSkelMemory(data.Dataset):
    def __init__(self, dataset_folder, split, categories=None, transform=None,
                 sampling=True, num_samples=4096,
                 return_surface=True, surface_sampling=True, pc_size=2048, replica=16,
                 return_skeleton=True,
                 skeleton_folder_basename='skeletons_min_sdf_iter_50',
                 data_subsample=None,
                 use_dilg_scale=True,
                 occupancies_base_folder='occupancies',
                 num_skel_samples=512,
                 use_fps=False):

        self.pc_size = pc_size
        self.use_dilg_scale = use_dilg_scale
        self.occupancies_base_folder = occupancies_base_folder
        self.num_skel_samples = num_skel_samples
        self.use_fps = use_fps

        self.transform = transform
        self.num_samples = num_samples
        self.sampling = sampling
        self.split = split

        self.dataset_folder = dataset_folder
        self.return_surface = return_surface
        self.return_skeleton = return_skeleton
        self.skeleton_folder_basename = skeleton_folder_basename
        self.surface_sampling = surface_sampling

        self.dataset_folder = dataset_folder

        # self.point_folder = os.path.join(self.dataset_folder, 'ShapeNetV2_point')
        # self.mesh_folder = os.path.join(self.dataset_folder, 'ShapeNetV2_watertight')

        self.point_folder = os.path.join(self.dataset_folder, '')
        self.mesh_folder = os.path.join(self.dataset_folder, '')

        if categories is None:
            categories = os.listdir(self.point_folder)
            categories = [c for c in categories if
                          os.path.isdir(os.path.join(self.point_folder, c)) and c.startswith('0')]
        categories.sort()
        print('Categories')
        print(categories)

        self.models = []
        for c_idx, c in enumerate(categories):
            subpath = os.path.join(self.point_folder, c)
            print(subpath)
            assert os.path.isdir(subpath)

            split_file = os.path.join(subpath, self.occupancies_base_folder, split + '.lst')
            with open(split_file, 'r') as f:
                models_c = f.read().split('\n')

            self.models += [
                {'category': c, 'model': m.replace('.npz', '')}
                for m in models_c
            ]

        self.replica = replica

        self.models = self.models[:data_subsample]
        print('Data size')
        print(len(self.models))
        print(self.models[:5])
        self.skeletons = []
        self.surfaces = []

        for idx in range(len(self.models)):


            category = self.models[idx]['category']
            model = self.models[idx]['model']

            point_path = os.path.join(self.point_folder, category, self.occupancies_base_folder, model + '.npz')
            if self.use_dilg_scale:
                with open(point_path.replace('.npz', '.npy'), 'rb') as f:
                    scale = np.load(f).item()
            else:
                scale = 1

            if self.return_surface:
                pc_path = os.path.join(self.mesh_folder, category, '4_pointcloud', model + '.npz')

                with np.load(pc_path) as data:
                    full_surface = data['points'].astype(np.float32)
                    #print(full_surface.shape)
                    full_surface = full_surface * scale
                    self.surfaces += [full_surface]

            if self.return_skeleton:
                skeleton_path = os.path.join(self.point_folder, category, self.skeleton_folder_basename,
                                             model + '_clean_skel.npz')
                # print(skeleton_path)
                full_skel = np.load(skeleton_path)['skel'] * scale
                self.skeletons += [full_skel]
                # print(skel.max(), surface.max())


    def __getitem__(self, idx):

        idx = idx % len(self.models)
        category = self.models[idx]['category']
        model = self.models[idx]['model']

        full_skel = self.skeletons[idx]
        skel = torch.from_numpy(full_skel).float()
        #print(skel.shape)
        if self.use_fps:
            skel, _ = fps_from_cloud(skel, N=self.num_skel_samples)
            skel = skel[0]
        else:
            skel_ind = np.random.default_rng().choice(skel.shape[0], self.num_skel_samples, replace=True)
            skel = skel[skel_ind]

        if self.return_surface:
            full_surface = self.surfaces[idx]
            ind = np.random.default_rng().choice(full_surface.shape[0], self.pc_size, replace=False)
            surface = full_surface[ind]
            surface = torch.from_numpy(surface)
        else:
            surface = torch.zeros(1)
        points, labels = torch.zeros(1), torch.zeros(1)

        if self.return_surface:
            return points, labels, surface, category_ids[category], skel
        else:
            return points, labels, category_ids[category], skel

    def __len__(self):
        if self.split != 'train':
            return len(self.models)
        else:
            return len(self.models) * self.replica