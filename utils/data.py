import numpy as np
import torch
import os
from collections import defaultdict

from torch.utils.data import Dataset
from .spatial import fps, get_dijkstra, fps_from_cloud, simple_fps
import json
from .io import get_graph
import trimesh
from sklearn.neighbors import KDTree
import h5py
from torchvision.transforms import RandomResizedCrop


class AxisScaling(object):
    def __init__(self, interval=(0.75, 1.25), jitter=True):
        assert isinstance(interval, tuple)
        self.interval = interval
        self.jitter = jitter

    def __call__(self, surface, skeleton):
        scaling = torch.rand(1, 3) * 0.5 + 0.75
        surface = surface * scaling
        skeleton = skeleton * scaling

        scale = (1 / torch.abs(surface).max().item()) * 0.49999999
        surface *= scale
        skeleton *= scale

        if self.jitter:
            surface += 0.0025 * torch.randn_like(surface)
            surface.clamp_(min=-1, max=1)

        return surface, skeleton


class AxisScalingCon(object):
    def __init__(self, interval=(0.75, 1.25), jitter=True):
        assert isinstance(interval, tuple)
        self.interval = interval
        self.jitter = jitter

    def __call__(self, surface):
        scaling = torch.rand(1, 3) * 0.1 + 0.95
        surface = surface * scaling.numpy()

        if self.jitter:
            surface += 0.0025 * np.random.randn(*surface.shape)

        return np.clip(surface, -0.52, 0.52)


class AxisScalingSkelray(object):
    def __init__(self, interval=(0.75, 1.25), jitter=True):
        assert isinstance(interval, tuple)
        self.interval = interval
        self.jitter = jitter

    def __call__(self, surface, skeleton, directions, disps, reg_ray_dists, reg_rays):
        scaling = np.random.uniform(0, 1, size=(1, 3)) * 0.5 + 0.75
        surface = surface * scaling
        skeleton = skeleton * scaling
        directions = directions * scaling
        dir_norms = np.linalg.norm(directions, axis=-1, keepdims=True)
        disps = disps * dir_norms[None, :, :]
        directions = directions / dir_norms

        reg_rays = reg_rays * scaling
        reg_rays_norms = np.linalg.norm(reg_rays, axis=-1, keepdims=True)
        reg_ray_dists = reg_ray_dists * reg_rays_norms
        reg_rays = reg_rays / reg_rays_norms

        scale = (1 / np.abs(surface).max().item()) * 0.499999999
        surface *= scale
        skeleton *= scale
        disps *= scale
        reg_ray_dists *= scale

        if self.jitter:
            surface += 0.0025 * torch.randn_like(surface)
            surface.clamp_(min=-0.5, max=0.5)

        return surface, skeleton, directions, disps, reg_ray_dists, reg_rays


def quantize_vals(vals, n_vals=256, shift=0.5, shape_scale=1.0):
    # print('quant vals', n_vals)
    delta = shape_scale / n_vals
    quant_vals = ((vals + shift) // delta).astype(np.int32)

    return quant_vals


def inv_quantize_vals(quant_vals, n_vals=256, shift=0.5, shape_scale=1.0):
    print('inv quant vals', n_vals)
    delta = shape_scale / n_vals
    vals = (quant_vals * delta - shift)

    return vals


def get_graph_skel(skel):
    verts = [tuple(item) for item in skel[:, 1:4]]
    verts += [tuple(item) for item in skel[:, 4:]]
    verts = list(set(verts))

    map_dict = dict(zip(verts, range(len(verts))))

    raw_edges = skel[:, 1:]
    all_edges = []
    for item in raw_edges:
        # print(item)
        vert1, vert2 = tuple(item[:3]), tuple(item[3:])
        edge = (map_dict[vert1], map_dict[vert2])
        all_edges += [edge]

    return verts, all_edges


def collate_reprs(data, select_random=True, return_skeleton=True):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """

    results = []

    for i, element in enumerate(data):
        if select_random:
            cur_ind = np.random.randint(0, len(element[0]), size=1)[0]

        result = [item[cur_ind] for i, item in enumerate(element) if i != 1]
        result = [result[0], element[1]] + result[1:]
        # print(len(result))
        results += [result]

    inputs = torch.stack([torch.FloatTensor(item[0]) for item in results], axis=0)
    try:
        sdfs = torch.stack([torch.FloatTensor(item[1]) for item in results], axis=0)
    except:
        sdfs = []
        for item in results:
            cur_sample = item[1]
            inds = np.random.randint(0, len(cur_sample), size=300000)
            cur_sample = cur_sample[inds]
            sdfs += [torch.FloatTensor(cur_sample)]
        sdfs = torch.stack(sdfs, axis=0)

    edges = [torch.LongTensor(item[2]) for item in results]
    outside_queries = [torch.FloatTensor(item[3]) for item in results]

    fin_result = [inputs, sdfs, edges, outside_queries]

    if return_skeleton:
        fin_result += [[torch.FloatTensor(item[4]) for item in results]]

    return fin_result

class SkeletonDataset(Dataset):

    def __init__(self, data_path, num_points=511, num_tokens=128, subsample=None,
                 block_size=1536, retrieval_key='surface_512',
                 ids_to_load=None, load_sdf=False, sdf_folder='sdf',
                 sdf_suffix='open3d'):
        if ids_to_load is not None:
            self.ids = ids_to_load
        else:
            self.ids = sorted(os.listdir(data_path))[:subsample]
        self.num_points = num_points
        self.block_size = block_size
        self.skeletons = []
        self.clouds = []
        self.tet_skeletons = []
        self.quant_clouds = []
        self.categories = []
        self.categories_list = sorted(['Bag', 'Bed', 'Bottle', 'Bowl', 'Chair', 'Clock',
                                       'Dishwasher', 'Display', 'Door', 'Earphone', 'Faucet',
                                       'Hat', 'Key', 'Keyboard', 'Knife', 'Lamp', 'Laptop',
                                       'Microwave', 'Mug', 'Refrigerator', 'Scissors',
                                       'StorageFurniture', 'Table', 'TrashCan', 'Vase'])
        self.categories_dict = dict(zip(self.categories_list, list(range(len(self.categories_list)))))
        self.fin_ids = []

        #np.random.shuffle(self.ids)
        self.num_base_tokens = num_tokens
        self.num_tokens = num_tokens + len(self.categories_list) + 1
        self.load_sdf = load_sdf

        if load_sdf:
            self.sdfs = []

        for i, item in enumerate(self.ids):
            print(i) if i == 0 or i % 100 == 99 else None
            try:
                tst = np.load(f'{data_path}/{item}/skeleton/model.npz')
                with open(f'{data_path}/{item}/meta.json') as f:
                    meta = json.load(f)
                clouds = tst[retrieval_key]
            except:
                print(f'Failed to load skeleton/json for shape {item}')
                continue

            if self.load_sdf:
                try:
                    load_path = f'{data_path}/{item}/{sdf_folder}/model_{sdf_suffix}.npz'
                   # print(load_path)
                    sdf_data = np.load(load_path)['sdf']
                except:
                    print(f'Failed to load sdf data for shape {item}')
            # sample = tst.sample(10000)
            # cloud = tst.sample(self.num_points)
            # cloud, inds = fps(sample, num_points)
            sorted_clouds = []
            for cloud in clouds:
                lexsort_inds = np.lexsort(cloud[:, [2, 0, 1]].T)
                sorted_clouds += [cloud[lexsort_inds]]

            sorted_clouds = np.stack(sorted_clouds, axis=0)

            if np.max(np.abs(sorted_clouds)) > 0.5:
                print(f'Shape {item} is not contained in unit cube, not loading it...')
                continue
            #print(sorted_clouds.max(axis=1).max(axis=0), sorted_clouds.min(axis=1).min(axis=0))
            self.clouds += [sorted_clouds]
            quantized_clouds = [quantize_vals(item, n_vals=self.num_tokens) for item in sorted_clouds]
            self.quant_clouds += [quantized_clouds]
            self.fin_ids += [item]
            self.categories += [meta['model_cat']]
            if self.load_sdf:
                self.sdfs += [sdf_data]


    def __len__(self):
        return len(self.clouds)

    def __getitem__(self, ind, select_random=True):
        # grab a chunk of (block_size + 1) characters from the data
        cloud = self.quant_clouds[ind]
        #print(self.fin_ids[ind])

        if select_random:
            cur_ind = np.random.randint(0, len(cloud), size=1)[0]
            cloud = cloud[cur_ind]
        # cloud = np.concatenate((np.array([[self.num_tokens, self.num_tokens, self.num_tokens]]), cloud), axis=0)
        # encode every character to an integer
        dix = cloud.reshape(-1)  # .reshape((-1, block_size))
        category_token = np.array([self.num_base_tokens + self.categories_dict[self.categories[ind]]])
        dix = np.concatenate((category_token, dix))
        #print(dix.max(), dix.min(), self.num_tokens)
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        result = [x,y]

        if self.load_sdf:
            result += [self.sdfs[ind]]

        return result



class SkeletonSDFDataset(Dataset):

    def __init__(self, data_path, num_points=511, num_tokens=128, subsample=None,
                 block_size=1536, retrieval_key='surface_512',
                 ids_to_load=None, load_sdf=False, sdf_folder='sdf',
                 sdf_suffix='open3d', load_edges=False,
                 load_skeletons=False,
                 load_suffix='',
                 subsample_cloud=None,
                 skeleton_key='skeleton_512'):
        if ids_to_load is not None:
            self.ids = ids_to_load[:subsample]
        else:
            self.ids = sorted(os.listdir(data_path))[:subsample]
        self.num_points = num_points
        self.block_size = block_size
        self.skeletons = []
        self.clouds = []
        self.tet_skeletons = []
        self.quant_clouds = []
        self.categories = []
        self.categories_list = sorted(['Bag', 'Bed', 'Bottle', 'Bowl', 'Chair', 'Clock',
                                       'Dishwasher', 'Display', 'Door', 'Earphone', 'Faucet',
                                       'Hat', 'Key', 'Keyboard', 'Knife', 'Lamp', 'Laptop',
                                       'Microwave', 'Mug', 'Refrigerator', 'Scissors',
                                       'StorageFurniture', 'Table', 'TrashCan', 'Vase'])
        self.categories_dict = dict(zip(self.categories_list, list(range(len(self.categories_list)))))
        self.fin_ids = []

        #np.random.shuffle(self.ids)
        self.num_base_tokens = num_tokens
        self.num_tokens = num_tokens + len(self.categories_list) + 1
        self.load_sdf = load_sdf
        self.load_edges = load_edges
        self.load_skeletons = load_skeletons
        self.subsample_cloud = subsample_cloud

        if load_sdf:
            self.sdfs = []

        if self.load_edges:
            #assert retrieval_key == 'tet_graph_points', ValueError('retrieval_key must be tet_graph_points to load edges')

            self.edges = []
            self.outside_queries = []

        for i, item in enumerate(self.ids):
            print(i) if i == 0 or i % 100 == 99 else None
            try:
                tst = np.load(f'{data_path}/{item}/skeleton/model{load_suffix}.npz', allow_pickle=True)
                with open(f'{data_path}/{item}/meta.json') as f:
                    meta = json.load(f)
                clouds = tst[retrieval_key]

            except:
                print(f'Failed to load skeleton/json for shape {item}')
                continue

            if self.load_sdf:
                try:
                    load_path = f'{data_path}/{item}/{sdf_folder}/model_{sdf_suffix}.npz'
                   # print(load_path)
                    sdf_data = np.load(load_path)['sdf']
                except:
                    print(f'Failed to load sdf data for shape {item}')

            if self.load_edges:
                edges = tst['tet_graph_edges']
                outside_queries = tst['tet_graph_outside_queries']

            if self.load_skeletons:
                skeletons = tst[skeleton_key]

            # sample = tst.sample(10000)
            # cloud = tst.sample(self.num_points)
            # cloud, inds = fps(sample, num_points)

            if np.max(np.abs(clouds)) > 0.5:
                print(f'Shape {item} is not contained in unit cube, not loading it...')
                continue
            #print(sorted_clouds.max(axis=1).max(axis=0), sorted_clouds.min(axis=1).min(axis=0))
            self.clouds += [clouds]

            if self.load_edges:
                self.edges += [edges]
                self.outside_queries += [outside_queries]
            self.fin_ids += [item]
            self.categories += [meta['model_cat']]
            if self.load_sdf:
                self.sdfs += [sdf_data]
            if self.load_skeletons:
                self.skeletons += [skeletons]

    def __len__(self):
        return len(self.clouds)

    def __getitem__(self, ind):

        if self.subsample_cloud is not None:
            sel_inds = np.random.randint(0, self.clouds[ind].shape[1], size=self.subsample_cloud)
            cloud = self.clouds[ind][:, sel_inds, :]
        else:
            cloud = self.clouds[ind]

        result = [cloud]

        if self.load_sdf:
            result += [self.sdfs[ind]]

        if self.load_edges:
            result += [self.edges[ind]]
            result += [self.outside_queries[ind]]

        if self.load_skeletons:
            result += [self.skeletons[ind]]

        return result

    def resample_full_skeletons(self, num_upsample=2048, clouds=True):
        if clouds:
            print(f'Resampling self.clouds to {num_upsample}')
            for i, item in enumerate(self.clouds):
                cur_cloud, _ = fps(item, num_upsample)
                self.clouds[i] = np.expand_dims(cur_cloud, axis=0)
        else:
            print(f'Resampling self.skeletons to {num_upsample}')
            for i, item in enumerate(self.skeletons):
                cur_skeleton, _ = fps(item, num_upsample)
                self.skeletons[i] = np.expand_dims(cur_skeleton, axis=0)



class SkeletonDatasetCombined(Dataset):

    def __init__(self, data_path, num_points=511, num_tokens=128, subsample=None,
                 block_size=1536, retrieval_key='surface_512',
                 ids_to_load=None, load_sdf=False, sdf_folder='sdf',
                 sdf_suffix='open3d', num_skeleton_points=256, num_surface_points=512,
                 load_suffix='',
                 load_full_skeleton=False):
        if ids_to_load is not None:
            self.ids = ids_to_load[:subsample]
        else:
            self.ids = sorted(os.listdir(data_path))[:subsample]
        self.num_points = num_points
        self.block_size = block_size
        self.quant_clouds_skeleton = []
        self.quant_clouds_surface = []
        self.clouds_skeleton = []
        self.clouds_surface = []
        self.clouds = []
        self.tet_skeletons = []
        self.quant_clouds = []
        self.categories = []
        self.categories_list = sorted(['Bag', 'Bed', 'Bottle', 'Bowl', 'Chair', 'Clock',
                                       'Dishwasher', 'Display', 'Door', 'Earphone', 'Faucet',
                                       'Hat', 'Key', 'Keyboard', 'Knife', 'Lamp', 'Laptop',
                                       'Microwave', 'Mug', 'Refrigerator', 'Scissors',
                                       'StorageFurniture', 'Table', 'TrashCan', 'Vase'])
        self.categories_dict = dict(zip(self.categories_list, list(range(len(self.categories_list)))))
        self.fin_ids = []

        # np.random.shuffle(self.ids)
        self.num_base_tokens = num_tokens
        self.num_tokens = num_tokens + len(self.categories_list) + 2
        self.load_sdf = load_sdf
        self.num_surface_points = 512
        self.num_skeleton_points = 256

        if load_sdf:
            self.sdfs = []

        for i, item in enumerate(self.ids):
            print(i) if i == 0 or i % 100 == 99 else None
            try:
                tst = np.load(f'{data_path}/{item}/skeleton/model{load_suffix}.npz')
                with open(f'{data_path}/{item}/meta.json') as f:
                    meta = json.load(f)
                if load_full_skeleton:
                    clouds_skeleton = tst[f'skeleton_full']
                    #print(clouds_skeleton.shape)
                else:
                    clouds_skeleton = tst[f'skeleton_{num_skeleton_points}']
                clouds_surface = tst[f'surface_{num_surface_points}']
            except:
                print(f'Failed to load skeleton/json for shape {item}')
                continue

            if self.load_sdf:
                try:
                    load_path = f'{data_path}/{item}/{sdf_folder}/model_{sdf_suffix}.npz'
                    # print(load_path)
                    sdf_data = np.load(load_path)['sdf']
                except:
                    print(f'Failed to load sdf data for shape {item}')
            # sample = tst.sample(10000)
            # cloud = tst.sample(self.num_points)
            # cloud, inds = fps(sample, num_points)
            sorted_clouds_skeleton, sorted_clouds_surface = [], []

            if not load_full_skeleton:
                for cloud in clouds_skeleton:
                    lexsort_inds = np.lexsort(cloud[:, [2, 0, 1]].T)
                    sorted_clouds_skeleton += [cloud[lexsort_inds]]
            else:
                lexsort_inds = np.lexsort(clouds_skeleton[:, [2, 0, 1]].T)
                sorted_clouds_skeleton += [clouds_skeleton[lexsort_inds]]

            for cloud in clouds_surface:
                lexsort_inds = np.lexsort(cloud[:, [2, 0, 1]].T)
                sorted_clouds_surface += [cloud[lexsort_inds]]

            if (np.max(np.abs(sorted_clouds_surface)) > 0.5) or (np.max(np.abs(sorted_clouds_skeleton)) > 0.5):
                print(f'Shape {item} is not contained in unit cube, not loading it...')
                continue

            #sorted_clouds_skeleton = np.stack(sorted_clouds_skeleton, axis=0)
            self.clouds_skeleton += [sorted_clouds_skeleton]
            sorted_clouds_surface = np.stack(sorted_clouds_surface, axis=0)
            self.clouds_surface += [sorted_clouds_surface]
            # print(self.num_tokens)
            quantized_clouds_skeleton = [quantize_vals(item, n_vals=self.num_base_tokens) for item in
                                         sorted_clouds_skeleton]
            self.quant_clouds_skeleton += [quantized_clouds_skeleton]
            quantized_clouds_surface = [quantize_vals(item, n_vals=self.num_base_tokens) for item in
                                        sorted_clouds_surface]
            self.quant_clouds_surface += [quantized_clouds_surface]

            self.fin_ids += [item]
            self.categories += [meta['model_cat']]

            if self.load_sdf:
                self.sdfs += [sdf_data]

    def __len__(self):
        return len(self.clouds_skeleton)

    def __getitem__(self, ind, select_random=False):
        # grab a chunk of (block_size + 1) characters from the data
        cloud_skeleton = self.quant_clouds_skeleton[ind]
        cloud_surface = self.quant_clouds_surface[ind]

        if select_random:
            cur_ind1 = np.random.randint(0, len(cloud_skeleton), size=1)[0]
            cur_ind2 = np.random.randint(0, len(cloud_surface), size=1)[0]
        else:
            cur_ind1 = 0
            cur_ind2 = 0

        cloud_skeleton = cloud_skeleton[cur_ind1]
        cloud_surface = cloud_surface[cur_ind2]

        # cloud = np.concatenate((np.array([[self.num_tokens, self.num_tokens, self.num_tokens]]), cloud), axis=0)
        # encode every character to an integer
        cloud1 = cloud_skeleton.reshape(-1)
        cloud2 = cloud_surface.reshape(-1)

        split_token = self.num_tokens - 1
        dix = np.concatenate((cloud1, np.array([split_token] * 3), cloud2))
        # dix = cloud.reshape(-1)#.reshape((-1, block_size))
        category_token = np.array([self.num_base_tokens + self.categories_dict[self.categories[ind]]])
        # print(category_token)
        dix = np.concatenate((category_token, dix))
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y


class Surf2SkeletonShapeNet(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, split, categories=None, transform=None,
                 sampling=True, num_samples=4096, return_surface=True,
                 surface_sampling=True, pc_size=2048,
                 point_folder_basename='',
                 mesh_folder_basename='',
                 occupancies_folder='occupancies',
                 subsample=None,
                 is_compact=False,
                 categories_to_use=None,
                 load_occupancies=False,
                 load_skeletons=False,
                 skeleton_folder_basename=''):

        self.pc_size = pc_size
        print(f'Occupancies folder is {occupancies_folder}')
        self.occupancies_folder = occupancies_folder

        self.transform = transform
        self.load_occupancies = load_occupancies
        self.load_skeletons = load_skeletons
        self.num_samples = num_samples
        print('Num samples is ', self.num_samples)
        self.sampling = sampling
        self.split = split
        self.compact_mult = 1 + 9 * is_compact
        print(f'Compact mult is {self.compact_mult}')

        self.dataset_folder = dataset_folder
        self.return_surface = return_surface
        self.surface_sampling = surface_sampling

        self.dataset_folder = dataset_folder
        self.point_folder = os.path.join(self.dataset_folder, point_folder_basename)
        self.mesh_folder = os.path.join(self.dataset_folder, mesh_folder_basename)
        self.skeleton_folder = os.path.join(self.dataset_folder, skeleton_folder_basename)
        self.skeleton_folder_basename = skeleton_folder_basename

        if categories is None:
            categories = os.listdir(self.point_folder)
            categories = [c for c in categories if
                          os.path.isdir(os.path.join(self.point_folder, c)) and c.startswith('0')]

        categories.sort()
        print(categories)

        self.models = []
        for c_idx, c in enumerate(categories):
            subpath = os.path.join(self.point_folder, c, self.occupancies_folder)
            # print(subpath)
            assert os.path.isdir(subpath)

            split_file = os.path.join(subpath, split + '.lst')
            with open(split_file, 'r') as f:
                models_c = f.read().split('\n')

            self.models += [
                {'category': c, 'model': m.replace('.npz', '')}
                for m in models_c
            ]
        print('Subsampling dataset')
        self.models = self.models[:subsample]

    def __getitem__(self, idx):
        category = self.models[idx]['category']
        model = self.models[idx]['model']

        point_path = os.path.join(self.point_folder, category, self.occupancies_folder, model + '.npz')

        if self.load_occupancies:
            try:
                with np.load(point_path) as data:
                    vol_points = data['vol_points'].astype(np.float32) * self.compact_mult
                    vol_label = data['vol_label']
                    near_points = data['near_points'].astype(np.float32) * self.compact_mult
                    near_label = data['near_label']
            except Exception as e:
                print(e)
                print(point_path)

            if self.sampling:
                ind = np.random.default_rng().choice(vol_points.shape[0], self.num_samples, replace=False)
                vol_points = vol_points[ind]
                vol_label = vol_label[ind]

                ind = np.random.default_rng().choice(near_points.shape[0], self.num_samples, replace=False)
                near_points = near_points[ind]
                near_label = near_label[ind]

            vol_points = torch.from_numpy(vol_points)
            vol_label = torch.from_numpy(vol_label).float()


        with open(point_path.replace('.npz', '.npy'), 'rb') as f:
            scale = np.load(f).item()

        if self.return_surface:
            try:
                pc_path = os.path.join(self.mesh_folder, category, '4_pointcloud', model + '.npz')
                with np.load(pc_path) as data:
                    surface = data['points'].astype(np.float32)
                    surface = surface# * scale
                if self.surface_sampling:
                    ind = np.random.default_rng().choice(surface.shape[0], self.pc_size, replace=False)
                    surface = surface[ind]
                surface = torch.from_numpy(surface)
            except:
                surface = None
        else:
            surface = None


        if self.load_occupancies:
            if self.split == 'train':
                near_points = torch.from_numpy(near_points)
                near_label = torch.from_numpy(near_label).float()

                points = torch.cat([vol_points, near_points], dim=0)
                labels = torch.cat([vol_label, near_label], dim=0)
            else:
                points = vol_points
                labels = vol_label
        else:
            points = None
            labels = None

        if self.load_skeletons:
            try:
                #print(category, self.skeleton_folder_basename, self.dataset_folder)
                skeleton_path = os.path.join(self.dataset_folder, category, self.skeleton_folder_basename, model + '_skel.txt')
                #print(skeleton_path)
                skel = np.loadtxt(skeleton_path)
                verts, edges = get_graph(skel)
                final_skeleton = torch.FloatTensor(verts), torch.LongTensor(edges)
            except:
                final_skeleton = None
        if self.transform:
            surface, points = self.transform(surface, points)

        return points, labels, surface, final_skeleton


    def __len__(self):
        return len(self.models)


class Surf2SkeletonShapeNetMemory(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, split, categories=None, transform=None,
                 sampling=True, num_samples=4096, return_surface=True,
                 surface_sampling=True, pc_size=2048,
                 point_folder_basename='',
                 data_folder_basename='',
                 mesh_folder_basename='watertight_simple',
                 occupancies_folder='occupancies',
                 simple_occ_folder='occ_simple',
                 subsample=None,
                 is_compact=False,
                 load_occupancies=False,
                 load_simple_occupancies=False,
                 load_skeletons=False,
                 skeleton_folder_basename='',
                 return_dijkstras=False,
                 load_normals=False,
                 load_correspondences=False,
                 load_meshes=False,
                 return_dict_getitem=False,
                 compute_corr_normals=False,
                 load_h5=False,
                 h5_folder='corr_data_128',
                 load_patches=True,
                 ids_to_load=None,
                 near_points_share=0.5,
                 load_skelrays=False,
                 skelray_folder=None,
                 load_reg_skelrays=False,
                 load_npz_skeletons=False,
                 occ_suffix='.npz',
                 check_points=False,
                 downsample_gt_skels=False,
                 gt_skel_sample=4096):

        if load_skelrays:
            assert skelray_folder is not None, "Skelray folder shouldn't be None"
        self.pc_size = pc_size
        print(f'Occupancies folder is {occupancies_folder}')
        self.load_npz_skeletons = load_npz_skeletons
        self.occupancies_folder = occupancies_folder
        self.load_correspondences = load_correspondences
        self.transform = transform
        self.load_occupancies = load_occupancies
        self.load_skeletons = load_skeletons
        self.load_normals = load_normals
        self.load_meshes = load_meshes
        self.load_h5 = load_h5
        self.load_patches = load_patches
        self.num_samples = num_samples
        self.compute_corr_normals = compute_corr_normals
        self.load_skelrays = load_skelrays
        self.load_reg_skelrays = load_reg_skelrays
        self.skelray_folder = skelray_folder
        print('Num samples is ', self.num_samples)
        self.sampling = sampling
        self.split = split
        self.compact_mult = 1 + 9 * is_compact
        self.return_dict_getitem = return_dict_getitem
        self.near_points_share = near_points_share

        print(f'Compact mult is {self.compact_mult}')

        self.dataset_folder = dataset_folder
        self.return_surface = return_surface
        self.surface_sampling = surface_sampling
        self.return_dijkstras = return_dijkstras
        self.h5_folder = h5_folder

        self.dataset_folder = dataset_folder
        self.point_folder = os.path.join(self.dataset_folder, point_folder_basename)
        self.data_folder = os.path.join(self.dataset_folder, data_folder_basename)
        self.mesh_folder_basename = mesh_folder_basename
        self.skeleton_folder = os.path.join(self.dataset_folder, skeleton_folder_basename)
        self.skeleton_folder_basename = skeleton_folder_basename
        self.clouds, self.skeletons, self.points, self.labels = [], [], [], []
        self.normals = []
        self.corrs = []
        self.meshes = []
        self.patches = []
        self.skelray_dicts = []
        self.occ_suffix = occ_suffix
        self.simple_occ_folder = simple_occ_folder
        self.load_simple_occupancies = load_simple_occupancies

        assert not (self.load_occupancies*self.load_simple_occupancies), "Can only load one type of occupancies"

        if categories is None:
            categories = os.listdir(self.point_folder)
            #print(categories)
            categories = [c for c in categories if
                          os.path.isdir(os.path.join(self.point_folder, c)) and c.startswith('0')]

        categories.sort()
        #print(categories)

        self.models = []
        if ids_to_load is None:

            for c_idx, c in enumerate(categories):
                subpath = os.path.join(self.point_folder, c, self.occupancies_folder)
                #print(subpath)
                assert os.path.isdir(subpath)

                split_file = os.path.join(subpath, split + '.lst')
                with open(split_file, 'r') as f:
                    models_c = f.read().split('\n')
                    models_c = [item for item in models_c if item != '']

                self.models += [
                    {'category': c, 'model': m.replace('.npz', '')}
                    for m in models_c
                ]
        else:
            self.models = ids_to_load
        #print(self.models)
        self.models = self.models[:subsample]
        self.clean_models = []

        for idx in range(len(self.models)):
            print(idx) if idx % 1000 == 0 else None
            load_func = self.load_item_h5 if self.load_h5 else self.load_item

            points, labels, surface, skeleton, normals, corr_dict, mesh, patches, skelray_dict = load_func(idx)

            if check_points and points is None:
                print(f"Couldn't load points for shape {self.models[idx]['model']}, idx {idx}")
                continue

            if ((skeleton is not None) and ((corr_dict is not None) \
                                           or (surface is not None) or (patches is not None)))\
                    or (skelray_dict is not None):
                self.clean_models += [self.models[idx]]
                if surface is not None:
                    self.clouds += [surface.type(torch.float16)]
                else:
                    self.clouds += [surface]

                if skeleton is not None:
                    if downsample_gt_skels:
                        fps_device = 'cuda' if torch.cuda.is_available() else 'cpu'
                        verts, _ = fps_from_cloud(torch.FloatTensor(skeleton[0]).to(fps_device), N=gt_skel_sample)
                        verts = verts[0].detach().cpu()
                    else:
                        verts = skeleton[0].type(torch.float32)
                    self.skeletons += [(verts, skeleton[1])]
                else:
                    self.skeletons += [skeleton]
                if points is not None:
                    self.points += [points.type(torch.float16)]
                else:
                    self.points += [points]
                self.labels += [labels]
                if normals is not None:
                    self.normals += [normals.type(torch.float16)]
                else:
                    self.normals += [normals]

                if patches is not None:
                    self.patches += [torch.FloatTensor(patches).type(torch.float16)]
                else:
                    self.patches += [patches]

                if skelray_dict is not None:
                    self.skelray_dicts += [skelray_dict]
                else:
                    self.skelray_dicts += [None]

                self.meshes += [mesh]
                self.corrs += [corr_dict]
            else:
                print(f"Couldn't load data for shape {self.models[idx]['model']}, idx {idx}")

        print('Finished loading, perecentage of good data is {:.3f}%.'.format(100*len(self.clean_models)/len(self.models)))

        if self.return_dijkstras:
            print('Getting dijkstras')
            self.get_dijkstras()

    def __getitem__(self, idx):

        if self.return_dijkstras:
            cur_dijkstra = self.skel_dijkstras[idx]
            #cur_dijkstra = get_dijkstra(*self.skeletons[idx])
        else:
            cur_dijkstra = None

        cur_cloud = self.clouds[idx]
        cur_skeleton = self.skeletons[idx]

        if self.transform is not None:
            new_surf, new_verts = self.transform(cur_cloud, cur_skeleton[0])
            new_skel = (new_verts, cur_skeleton[1])
        else:
            new_surf = cur_cloud
            new_skel = cur_skeleton

        values = self.points[idx], self.labels[idx], new_surf, new_skel, cur_dijkstra, self.normals[idx], \
                 self.corrs[idx], self.meshes[idx], self.patches[idx], self.skelray_dicts[idx]

        if not self.return_dict_getitem:
            return values
        else:
            keys = ['points', 'labels', 'cloud', 'skeleton', 'dijkstras',
                    'normals', 'corrs', 'mesh', 'patches', 'skelray_dict']
            return dict(zip(keys, values))

    def __len__(self):
        return len(self.clouds)

    def load_item_h5(self, idx):
        category = self.models[idx]['category']
        model = self.models[idx]['model']

        try:
            h5_path = os.path.join(self.point_folder, category, self.h5_folder, model + '.h5')
            with h5py.File(h5_path, 'r') as f:
                if self.load_patches:
                    patches = np.array(f['corr_patches'])
                else:
                    patches = None
                verts, edges = np.array(f['skel_verts']), np.array(f['skel_edges'])
                cloud = np.array(f['cloud_sample'])

            points = None
            labels = None
            surface = torch.FloatTensor(cloud)
            if len(verts) <= 6000:
                final_skeleton = (torch.FloatTensor(verts), torch.LongTensor(edges))
            else:
                print(f'Skeleton for shape {model} from category {category} has > 6000 vertices')
                final_skeleton = None
            corr_dict = None
            normals = None
            if self.load_meshes:
                try:
                    mesh_path = os.path.join(self.dataset_folder, category,
                                             self.mesh_folder_basename,
                                             model + '.off')
                    #print(mesh_path)
                    mesh = trimesh.load(mesh_path)
                    # print(skeleton_path)

                except:
                    mesh = None
            else:
                mesh = None
        except:
            return [None]*8

        if patches is not None and final_skeleton is not None:
            if len(patches) != len(final_skeleton[0]):
                print(f"Number of patches unequal to number of skeleton points for shape {model} in category {category}")
                patches = None
                final_skeleton = None

        return points, labels, surface, final_skeleton, normals, corr_dict, mesh, patches

    def load_item(self, idx):

        category = self.models[idx]['category']
        model = self.models[idx]['model']

        point_path = os.path.join(self.point_folder, category, self.occupancies_folder, model + '.npz')

        if self.load_simple_occupancies:
            simple_occ_path = os.path.join(self.point_folder, category, self.simple_occ_folder, model + '_queries.npz')
            try:
                #print('loading simple occs')
                with np.load(simple_occ_path) as data:
                    #print(list(data.keys()))
                    points = torch.FloatTensor(data['queries']).type(torch.float16)
                    labels = torch.from_numpy(data['labels']).type(torch.float16)
                    #print(points.shape, labels.shape)

            except Exception as e:
                #print(e)
                #print(point_path)
                points = None
                labels = None

        if self.load_occupancies:
            print('OCCS 1')
            try:
                with np.load(point_path) as data:
                    #print(list(data.keys()))
                    vol_points = data['vol_points'].astype(np.float32) * self.compact_mult
                    vol_label = data['vol_label']
                    near_points = data['near_points'].astype(np.float32) * self.compact_mult
                    near_label = data['near_label']
            except Exception as e:
                print(e)
                print(point_path)

            if self.sampling:
                num_near_points = int(self.num_samples * self.near_points_share)
                ind = np.random.default_rng().choice(vol_points.shape[0], self.num_samples - num_near_points, replace=False)
                vol_points = vol_points[ind]
                vol_label = vol_label[ind]

                ind = np.random.default_rng().choice(near_points.shape[0], num_near_points, replace=False)
                near_points = near_points[ind]
                near_label = near_label[ind]

                #print('Sampling check')
                #print(vol_points.shape, near_points.shape)

            vol_points = torch.from_numpy(vol_points)
            vol_label = torch.from_numpy(vol_label).float()


        with open(point_path.replace('.npz', '.npy'), 'rb') as f:
            scale = np.load(f).item()

        if self.return_surface:
            try:
                pc_path = os.path.join(self.data_folder, category, '4_pointcloud', model + '.npz')
                with np.load(pc_path) as data:
                    surface = data['points'].astype(np.float32)
                    surface = surface# * scale

                    if self.load_normals:
                        normals = data['normals'].astype(np.float32)
                    else:
                        normals = None

                if self.surface_sampling:
                    ind = np.random.default_rng().choice(surface.shape[0], self.pc_size, replace=False)
                    surface = surface[ind]

                    if normals is not None:
                        normals = normals[ind]

                surface = torch.from_numpy(surface)
                if normals is not None:
                    normals = torch.from_numpy(normals)
            except:
                surface = None
                normals = None
        else:
            surface = None
            normals = None


        if self.load_occupancies:
            if self.split == 'train' or self.split == 'val':
                near_points = torch.from_numpy(near_points)
                near_label = torch.from_numpy(near_label).float()

                points = torch.cat([vol_points, near_points], dim=0) / scale
                labels = torch.cat([vol_label, near_label], dim=0)
            else:
                points = vol_points / scale
                labels = vol_label
        elif not self.load_simple_occupancies:
            points = None
            labels = None
        else:
            None

        if self.load_skeletons:
            try:
                #print(category, self.skeleton_folder_basename, self.dataset_folder)
                if self.load_npz_skeletons:
                    skeleton_path = os.path.join(self.dataset_folder, category, self.skeleton_folder_basename,
                                                 model + '_clean_skel.npz')
                    # print(skeleton_path)
                    skel = np.load(skeleton_path)['skel']
                    #print(skel.shape)
                    if skel != []:
                        verts, edges = skel, None
                    # skel_verts, inds_skel = fps_from_cloud(full_verts, N=num_skel_sample)

                        final_skeleton = torch.FloatTensor(verts), edges
                    else:
                        final_skeleton = None
                else:
                    skeleton_path = os.path.join(self.dataset_folder, category, self.skeleton_folder_basename, model + '_skel.txt')
                    #print(skeleton_path)
                    skel = np.loadtxt(skeleton_path)
                    verts, edges = get_graph(skel)
                    #skel_verts, inds_skel = fps_from_cloud(full_verts, N=num_skel_sample)

                    final_skeleton = torch.FloatTensor(verts), torch.LongTensor(edges)
            except:
                final_skeleton = None
        else:
            final_skeleton = None

        if self.load_meshes:
            try:
                mesh_path = os.path.join(self.dataset_folder, category,
                                         self.mesh_folder_basename,
                                         model + '.off')
                #print(mesh_path)
                mesh = trimesh.load(mesh_path)
                # print(skeleton_path)

            except:
                print(f'Failed to load mesh for {mesh_path}')
                mesh = None
        else:
            mesh = None


        if self.load_correspondences:
            try:
                #print(category, self.skeleton_folder_basename, self.dataset_folder)
                corr_path = os.path.join(self.dataset_folder, category, self.skeleton_folder_basename, model + '_corr.txt')
                #print(skeleton_path)
                corr = np.loadtxt(corr_path)[:, 1:]
                corr_dict = defaultdict(list)

                if self.compute_corr_normals:
                    assert self.load_meshes, "load_meshes needs to be true get corr normals"
                    assert mesh is not None, "Mesh is None"
                    vertex_normals = trimesh.geometry.weighted_vertex_normals(len(mesh.vertices),
                                                                              mesh.faces,
                                                                              mesh.face_normals,
                                                                              mesh.face_angles)
                    keys = [tuple(item) for item in mesh.vertices.round(4)]
                    normal_map = dict(zip(keys, vertex_normals))
                    tree = KDTree(np.array(keys), leaf_size=40)


                for i in range(len(corr)):
                    #print(i)
                    surf_point = corr[i, 3:]
                    if self.compute_corr_normals:
                        normal_point = normal_map.get(tuple(surf_point.round(4)), None)
                        if normal_point is None:
                            #print('Querying', i)
                            ind = tree.query(surf_point.reshape(1,3), k=1)[1][0][0]
                            normal_point = normal_map[keys[ind]]
                        normal_point = tuple(normal_point)
                    else:
                        normal_point = tuple()
                    corr_dict[tuple(corr[i, :3])] += [tuple(surf_point) + normal_point]

                for key in corr_dict.keys():
                    if corr_dict[key]:
                        corr_dict[key] = np.array(corr_dict[key])

                for item in corr_dict.keys():
                    if len(item) < 3:
                        print(corr_path, item)
            except:
                corr_dict = None
        else:
            corr_dict = None

        if self.load_skelrays:

            skelray_path = os.path.join(self.point_folder, category, self.skelray_folder, model + '_skelrays.npz')
            try:
                skelray_file = np.load(skelray_path)

                skelrays, skelray_skel, directions = skelray_file['ray_dists'], skelray_file['skel'],\
                                                        skelray_file['directions']
                surf_mask = skelray_file['surf_mask']
                skelray_dict = {}
                skelray_dict['skelrays'] = skelrays
                skelray_dict['skel'] = skelray_skel
                skelray_dict['directions'] = directions
                skelray_dict['surf_mask'] = surf_mask
                if self.load_reg_skelrays:
                    skelray_dict['reg_ray_dists'] = skelray_file['reg_ray_dists']
                    skelray_dict['reg_rays'] = skelray_file['reg_rays']
                    skelray_dict['reg_skel_inds'] = skelray_file['reg_skel_inds']

            except:
                print('Failed to load skelrays from file', skelray_path)
                skelray_dict = None



        else:
           skelray_dict = None

        patches = None

        return points, labels, surface, final_skeleton, normals, corr_dict, mesh, patches,\
                skelray_dict

    def get_dijkstras(self):

        self.skel_dijkstras = [get_dijkstra(*item) for item in self.skeletons]


class Surf2SkeletonShapeNetSSD(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, split, categories=None, transform=None,
                 sampling=True, num_samples=4096, return_surface=True,
                 surface_sampling=True, pc_size=2048,
                 point_folder_basename='',
                 data_folder_basename='',
                 mesh_folder_basename='watertight_simple',
                 occupancies_folder='occupancies',
                 simple_occ_folder='occ_simple',
                 subsample=None,
                 is_compact=False,
                 load_occupancies=False,
                 load_simple_occupancies=False,
                 load_skeletons=False,
                 skeleton_folder_basename='',
                 return_dijkstras=False,
                 load_normals=False,
                 load_correspondences=False,
                 load_meshes=False,
                 return_dict_getitem=False,
                 compute_corr_normals=False,
                 load_h5=False,
                 h5_folder='corr_data_128',
                 load_patches=True,
                 ids_to_load=None,
                 near_points_share=0.5,
                 load_skelrays=False,
                 skelray_folder=None,
                 load_reg_skelrays=False,
                 load_npz_skeletons=False,
                 occ_suffix='.npz'):

        if load_skelrays:
            assert skelray_folder is not None, "Skelray folder shouldn't be None"
        self.pc_size = pc_size
        print(f'Occupancies folder is {occupancies_folder}')
        self.load_npz_skeletons = load_npz_skeletons
        self.occupancies_folder = occupancies_folder
        self.load_correspondences = load_correspondences
        self.transform = transform
        self.load_occupancies = load_occupancies
        self.load_skeletons = load_skeletons
        self.load_normals = load_normals
        self.load_meshes = load_meshes
        self.load_h5 = load_h5
        self.load_patches = load_patches
        self.num_samples = num_samples
        self.compute_corr_normals = compute_corr_normals
        self.load_skelrays = load_skelrays
        self.load_reg_skelrays = load_reg_skelrays
        self.skelray_folder = skelray_folder
        print('Num samples is ', self.num_samples)
        self.sampling = sampling
        self.split = split
        self.compact_mult = 1 + 9 * is_compact
        self.return_dict_getitem = return_dict_getitem
        self.near_points_share = near_points_share

        print(f'Compact mult is {self.compact_mult}')

        self.dataset_folder = dataset_folder
        self.return_surface = return_surface
        self.surface_sampling = surface_sampling
        self.return_dijkstras = return_dijkstras
        self.h5_folder = h5_folder

        self.dataset_folder = dataset_folder
        self.point_folder = os.path.join(self.dataset_folder, point_folder_basename)
        self.data_folder = os.path.join(self.dataset_folder, data_folder_basename)
        self.mesh_folder_basename = mesh_folder_basename
        self.skeleton_folder = os.path.join(self.dataset_folder, skeleton_folder_basename)
        self.skeleton_folder_basename = skeleton_folder_basename
        self.clouds, self.skeletons, self.points, self.labels = [], [], [], []
        self.normals = []
        self.corrs = []
        self.meshes = []
        self.patches = []
        self.skelray_dicts = []
        self.occ_suffix = occ_suffix
        self.simple_occ_folder = simple_occ_folder
        self.load_simple_occupancies = load_simple_occupancies

        assert not (self.load_occupancies*self.load_simple_occupancies), "Can only load one type of occupancies"

        if categories is None:
            categories = os.listdir(self.point_folder)
            #print(categories)
            categories = [c for c in categories if
                          os.path.isdir(os.path.join(self.point_folder, c)) and c.startswith('0')]

        categories.sort()
        #print(categories)

        self.models = []
        if ids_to_load is None:

            for c_idx, c in enumerate(categories):
                subpath = os.path.join(self.point_folder, c, self.occupancies_folder)
                #print(subpath)
                assert os.path.isdir(subpath)

                split_file = os.path.join(subpath, split + '.lst')
                with open(split_file, 'r') as f:
                    models_c = f.read().split('\n')

                self.models += [
                    {'category': c, 'model': m.replace('.npz', '')}
                    for m in models_c
                ]
        else:
            self.models = ids_to_load
        #print(self.models)
        self.models = self.models[:subsample]
        self.clean_models = []

        #def load_item():
        #    for idx in range(len(self.models)):
        #        #print(idx) if idx % 1000 == 0 else None

        #print('Finished loading, perecentage of good data is {:.3f}%.'.format(100*len(self.clean_models)/len(self.models)))

        if self.return_dijkstras:
            print('Getting dijkstras')
            self.get_dijkstras()

    def __getitem__(self, idx):

        load_func = self.load_item_h5 if self.load_h5 else self.load_item

        points, labels, surface, skeleton, normals, corr_dict, mesh, patches, skelray_dict = load_func(idx)

        if ((skeleton is not None) and ((corr_dict is not None) \
                                        or (surface is not None) or (patches is not None))) \
                or (skelray_dict is not None):
            #self.clean_models += [self.models[idx]]
            if surface is not None:
                result_clouds = surface.type(torch.float32)
            else:
                result_clouds = surface

            if skeleton is not None:
                result_skeletons = (skeleton[0].type(torch.float16), skeleton[1])
            else:
                result_skeletons = skeleton
            if points is not None:
                result_points = points.type(torch.float32)
            else:
                result_points = points
            result_labels = labels
            if normals is not None:
                result_normals = normals.type(torch.float16)
            else:
                result_normals = normals

            if patches is not None:
                result_patches = torch.FloatTensor(patches).type(torch.float32)
            else:
                result_patches = patches

            if skelray_dict is not None:
                result_skelray_dicts = skelray_dict
            else:
                result_skelray_dicts = None

            result_meshes = mesh
            result_corrs = corr_dict

        values = result_points, result_labels, result_clouds, result_skeletons, \
                 None, result_normals, \
                 result_corrs, result_meshes, result_patches, result_skelray_dicts

        if not self.return_dict_getitem:
            return values
        else:
            keys = ['points', 'labels', 'cloud', 'skeleton', 'dijkstras',
                    'normals', 'corrs', 'mesh', 'patches', 'skelray_dict']
            return dict(zip(keys, values))

    def __len__(self):
        return len(self.models)

    def load_item_h5(self, idx):
        category = self.models[idx]['category']
        model = self.models[idx]['model']

        try:
            h5_path = os.path.join(self.point_folder, category, self.h5_folder, model + '.h5')
            with h5py.File(h5_path, 'r') as f:
                if self.load_patches:
                    patches = np.array(f['corr_patches'])
                else:
                    patches = None
                verts, edges = np.array(f['skel_verts']), np.array(f['skel_edges'])
                cloud = np.array(f['cloud_sample'])

            points = None
            labels = None
            surface = torch.FloatTensor(cloud)
            if len(verts) <= 6000:
                final_skeleton = (torch.FloatTensor(verts), torch.LongTensor(edges))
            else:
                print(f'Skeleton for shape {model} from category {category} has > 6000 vertices')
                final_skeleton = None
            corr_dict = None
            normals = None
            if self.load_meshes:
                try:
                    mesh_path = os.path.join(self.dataset_folder, category,
                                             self.mesh_folder_basename,
                                             model + '.off')
                    #print(mesh_path)
                    mesh = trimesh.load(mesh_path)
                    # print(skeleton_path)

                except:
                    mesh = None
            else:
                mesh = None
        except:
            return [None]*8

        if patches is not None and final_skeleton is not None:
            if len(patches) != len(final_skeleton[0]):
                print(f"Number of patches unequal to number of skeleton points for shape {model} in category {category}")
                patches = None
                final_skeleton = None

        return points, labels, surface, final_skeleton, normals, corr_dict, mesh, patches

    def load_item(self, idx):

        category = self.models[idx]['category']
        model = self.models[idx]['model']

        point_path = os.path.join(self.point_folder, category, self.occupancies_folder, model + '.npz')

        if self.load_simple_occupancies:
            simple_occ_path = os.path.join(self.point_folder, category, self.simple_occ_folder, model + '_queries.npz')
            try:
                #print('loading simple occs')
                with np.load(simple_occ_path) as data:
                    #print(list(data.keys()))
                    points = torch.FloatTensor(data['queries']).type(torch.float16)
                    labels = torch.from_numpy(data['labels']).type(torch.float16)
                    #print(points.shape, labels.shape)

            except Exception as e:
                #print(e)
                #print(point_path)
                points = None
                labels = None

        if self.load_occupancies:
            try:
                with np.load(point_path) as data:
                    #print(list(data.keys()))
                    vol_points = data['vol_points'].astype(np.float32) * self.compact_mult
                    vol_label = data['vol_label']
                    near_points = data['near_points'].astype(np.float32) * self.compact_mult
                    near_label = data['near_label']
            except Exception as e:
                print(e)
                print(point_path)

            if self.sampling:
                num_near_points = int(self.num_samples * self.near_points_share)
                ind = np.random.default_rng().choice(vol_points.shape[0], self.num_samples - num_near_points, replace=False)
                vol_points = vol_points[ind]
                vol_label = vol_label[ind]

                ind = np.random.default_rng().choice(near_points.shape[0], num_near_points, replace=False)
                near_points = near_points[ind]
                near_label = near_label[ind]

                #print('Sampling check')
                #print(vol_points.shape, near_points.shape)

            vol_points = torch.from_numpy(vol_points)
            vol_label = torch.from_numpy(vol_label).float()


        with open(point_path.replace('.npz', '.npy'), 'rb') as f:
            scale = np.load(f).item()

        if self.return_surface:
            try:
                pc_path = os.path.join(self.data_folder, category, '4_pointcloud', model + '.npz')
                with np.load(pc_path) as data:
                    surface = data['points'].astype(np.float32)
                    surface = surface# * scale

                    if self.load_normals:
                        normals = data['normals'].astype(np.float32)
                    else:
                        normals = None

                if self.surface_sampling:
                    ind = np.random.default_rng().choice(surface.shape[0], self.pc_size, replace=False)
                    surface = surface[ind]

                    if normals is not None:
                        normals = normals[ind]

                surface = torch.from_numpy(surface)
                if normals is not None:
                    normals = torch.from_numpy(normals)
            except:
                surface = None
                normals = None
        else:
            surface = None
            normals = None


        if self.load_occupancies:
            if self.split == 'train' or self.split == 'val':
                near_points = torch.from_numpy(near_points)
                near_label = torch.from_numpy(near_label).float()

                points = torch.cat([vol_points, near_points], dim=0) / scale
                labels = torch.cat([vol_label, near_label], dim=0)
                #print('Check')
                #print(points.shape)
            else:
                points = vol_points / scale
                labels = vol_label
        elif not self.load_simple_occupancies:
            points = None
            labels = None
        else:
            None

        if self.load_skeletons:
            try:
                #print(category, self.skeleton_folder_basename, self.dataset_folder)
                if self.load_npz_skeletons:
                    skeleton_path = os.path.join(self.dataset_folder, category, self.skeleton_folder_basename,
                                                 model + '_clean_skel.npz')
                    # print(skeleton_path)
                    skel = np.load(skeleton_path)['skel']
                    #print(skel.shape)
                    if skel != []:
                        verts, edges = skel, None
                    # skel_verts, inds_skel = fps_from_cloud(full_verts, N=num_skel_sample)

                        final_skeleton = torch.FloatTensor(verts), edges
                    else:
                        final_skeleton = None
                else:
                    skeleton_path = os.path.join(self.dataset_folder, category, self.skeleton_folder_basename, model + '_skel.txt')
                    #print(skeleton_path)
                    skel = np.loadtxt(skeleton_path)
                    verts, edges = get_graph(skel)
                    #skel_verts, inds_skel = fps_from_cloud(full_verts, N=num_skel_sample)

                    final_skeleton = torch.FloatTensor(verts), torch.LongTensor(edges)
            except:
                final_skeleton = None
        else:
            final_skeleton = None

        if self.load_meshes:
            try:
                mesh_path = os.path.join(self.dataset_folder, category,
                                         self.mesh_folder_basename,
                                         model + '.off')
                #print(mesh_path)
                mesh = trimesh.load(mesh_path)
                # print(skeleton_path)

            except:
                print(f'Failed to load mesh for {mesh_path}')
                mesh = None
        else:
            mesh = None


        if self.load_correspondences:
            try:
                #print(category, self.skeleton_folder_basename, self.dataset_folder)
                corr_path = os.path.join(self.dataset_folder, category, self.skeleton_folder_basename, model + '_corr.txt')
                #print(skeleton_path)
                corr = np.loadtxt(corr_path)[:, 1:]
                corr_dict = defaultdict(list)

                if self.compute_corr_normals:
                    assert self.load_meshes, "load_meshes needs to be true get corr normals"
                    assert mesh is not None, "Mesh is None"
                    vertex_normals = trimesh.geometry.weighted_vertex_normals(len(mesh.vertices),
                                                                              mesh.faces,
                                                                              mesh.face_normals,
                                                                              mesh.face_angles)
                    keys = [tuple(item) for item in mesh.vertices.round(4)]
                    normal_map = dict(zip(keys, vertex_normals))
                    tree = KDTree(np.array(keys), leaf_size=40)


                for i in range(len(corr)):
                    #print(i)
                    surf_point = corr[i, 3:]
                    if self.compute_corr_normals:
                        normal_point = normal_map.get(tuple(surf_point.round(4)), None)
                        if normal_point is None:
                            #print('Querying', i)
                            ind = tree.query(surf_point.reshape(1,3), k=1)[1][0][0]
                            normal_point = normal_map[keys[ind]]
                        normal_point = tuple(normal_point)
                    else:
                        normal_point = tuple()
                    corr_dict[tuple(corr[i, :3])] += [tuple(surf_point) + normal_point]

                for key in corr_dict.keys():
                    if corr_dict[key]:
                        corr_dict[key] = np.array(corr_dict[key])

                for item in corr_dict.keys():
                    if len(item) < 3:
                        print(corr_path, item)
            except:
                corr_dict = None
        else:
            corr_dict = None

        if self.load_skelrays:

            skelray_path = os.path.join(self.point_folder, category, self.skelray_folder, model + '_skelrays.npz')
            try:
                skelray_file = np.load(skelray_path)

                skelrays, skelray_skel, directions = skelray_file['ray_dists'], skelray_file['skel'],\
                                                        skelray_file['directions']
                surf_mask = skelray_file['surf_mask']
                skelray_dict = {}
                skelray_dict['skelrays'] = skelrays
                skelray_dict['skel'] = skelray_skel
                skelray_dict['directions'] = directions
                skelray_dict['surf_mask'] = surf_mask
                if self.load_reg_skelrays:
                    skelray_dict['reg_ray_dists'] = skelray_file['reg_ray_dists']
                    skelray_dict['reg_rays'] = skelray_file['reg_rays']
                    skelray_dict['reg_skel_inds'] = skelray_file['reg_skel_inds']

            except:
                print('Failed to load skelrays from file', skelray_path)
                skelray_dict = None



        else:
           skelray_dict = None

        if self.transform:
            surface, points = self.transform(surface, points)

        patches = None

        return points, labels, surface, final_skeleton, normals, corr_dict, mesh, patches,\
                skelray_dict

    def get_dijkstras(self):

        self.skel_dijkstras = [get_dijkstra(*item) for item in self.skeletons]



def collate_reprs_skel(data, num_surface_sample=2048, num_skel_sample=2048, return_occupancies=False,
                      occ_sample=1024, return_full_graphs=False, return_dijkstras=False,
                       return_normals=False):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """

    all_clouds = []
    all_skeletons = []
    if return_occupancies:
        all_queries, all_labels = [], []

    if return_full_graphs:
        full_graphs = []

    if return_dijkstras:
        all_dijkstras = []

    if return_normals:
        all_normals = []


    for i, cur_data in enumerate(data):

        cur_queries, cur_occupancies, cur_cloud, cur_skeleton = cur_data[:4]

        full_verts = cur_skeleton[0]
        skel_edges = cur_skeleton[1]
        inds_cloud = np.random.randint(0, cur_cloud.shape[0], size=num_surface_sample)
        cur_cloud = cur_cloud[inds_cloud].unsqueeze(0)
        #inds_skel = np.random.randint(0, full_verts.shape[0], size=num_skel_sample)
        skel_verts, inds_skel = fps_from_cloud(full_verts, N=num_skel_sample)
        inds_skel = inds_skel[0]
        #skel_verts = sample_points_from_edges(full_verts, skel_edges, num_skel_sample)
        #skel_verts = full_verts[inds_skel]
        #skel_verts = skel_verts.unsqueeze(0)
        all_clouds += [cur_cloud]
        all_skeletons += [skel_verts]

        if return_occupancies:
            inds_occ = np.random.randint(0, cur_occupancies.shape[0], size=occ_sample)
            sampled_queries = cur_queries[inds_occ].unsqueeze(0)
            sampled_occupancies = cur_occupancies[inds_occ].unsqueeze(0)
            all_queries += [sampled_queries]
            all_labels += [sampled_occupancies]

        if return_full_graphs:
            full_graphs += [cur_skeleton]

        if return_dijkstras:
            cur_dijkstra = cur_data[4]
            cur_dijkstra = cur_dijkstra[inds_skel, :][:, inds_skel]
            all_dijkstras += [cur_dijkstra]

        if return_normals:
            cur_normals = cur_data[5]
            cur_normals = cur_normals[inds_cloud].unsqueeze(0)
            all_normals += [cur_normals]


    all_clouds = torch.cat(all_clouds, axis=0).type(torch.float32)
    all_skeletons = torch.cat(all_skeletons, axis=0).type(torch.float32)

    result = [all_clouds, all_skeletons]

    if return_occupancies:
        all_queries = torch.cat(all_queries, axis=0).type(torch.float32)
        all_labels = torch.cat(all_labels, axis=0).type(torch.float32)
        result += [all_queries, all_labels]

    if return_full_graphs:
        result += [full_graphs]

    if return_dijkstras:
        result += [all_dijkstras]

    if return_normals:
        all_normals = torch.cat(all_normals, axis=0).type(torch.float32)
        result += [all_normals]

    return result


def jitter_skeleton_supervision(skels, rays, scales, skel_ids,
                                reg_ray_scales, reg_rays, reg_ray_ids,
                                jitter_scale=0.005,
                                max_disps=None):
    if max_disps is not None:
        jitter_scales = torch.minimum(max_disps * 0.5, torch.ones_like(max_disps).float() * jitter_scale)

    else:
        jitter_scales = torch.ones_like(skels) * jitter_scale
    # print('Jitter scales', jitter_scales.shape, jitter_scales.min(), jitter_scales.max())
    skel_jitter = torch.randn_like(skels) * jitter_scales
    jittered_skels = skels + skel_jitter

    sel_pts = skels[torch.arange(skels.size(0)).unsqueeze(1), skel_ids]
    jittered_skel_pts = jittered_skels[torch.arange(skels.size(0)).unsqueeze(1), skel_ids]
    surface_pts = sel_pts + rays * scales
    jittered_disps = surface_pts - jittered_skel_pts
    jittered_scales = torch.linalg.norm(jittered_disps, axis=-1, keepdim=True)
    jittered_rays = jittered_disps / jittered_scales

    sel_reg_skel_pts = skels[torch.arange(skels.size(0)).unsqueeze(1), reg_ray_ids.reshape(len(skels), -1)]
    sel_jittered_reg_pts = jittered_skels[torch.arange(skels.size(0)).unsqueeze(1), reg_ray_ids.reshape(len(skels), -1)]
    flat_reg_ray_scales = reg_ray_scales.reshape(len(skels), -1, 1)
    reg_surf_pts = sel_reg_skel_pts + flat_reg_ray_scales * reg_rays.reshape(len(skels), -1, 3)

    jittered_reg_disps = reg_surf_pts - sel_jittered_reg_pts
    jittered_reg_scales = torch.linalg.norm(jittered_reg_disps, axis=-1, keepdim=True)

    jittered_reg_rays = jittered_reg_disps / jittered_reg_scales

    jittered_reg_rays = jittered_reg_rays.reshape(*reg_rays.shape)
    jittered_reg_scales = jittered_reg_scales.reshape(*reg_ray_scales.shape)

    return jittered_skels, jittered_rays, jittered_scales, skel_ids, jittered_reg_scales, jittered_reg_rays, reg_ray_ids


def collate_reprs_skelrays(data, num_surface_sample=2048, num_ray_queries=1000,
                           num_skel_samples=512, subsample_skels=False,
                           non_surface_weight=0.1, return_reg_data=False,
                           num_reg_sample=1000,
                           return_surface_queries=False,
                           num_surface_queries=1000,
                           use_fps_sampling=False,
                           fps_device='cuda',
                           unpack_bits=False,
                           transform=None,
                           sample_proportional_to_disp=False,
                           jitter_skel_supervision=False,
                           jitter_scale=0.005):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """

    all_clouds = []
    all_skeletons = []
    all_rays = []
    all_skel_ids = []
    all_scales = []
    if jitter_skel_supervision:
        all_max_disps = []

    if return_surface_queries:
        all_surface_queries = []

    if return_reg_data:
        all_reg_ray_dists = []
        all_reg_rays = []
        all_reg_skel_inds = []

    for i, cur_data in enumerate(data):

        full_cloud = cur_data['cloud']
        inds_cloud = np.random.randint(0, full_cloud.shape[0], size=num_surface_sample)
        cur_cloud = full_cloud[inds_cloud].unsqueeze(0)

        if return_surface_queries:
            inds_surface = np.random.randint(0, full_cloud.shape[0], size=num_surface_queries)
            cur_surface_queries = full_cloud[inds_surface].unsqueeze(0)

        cur_dict = cur_data['skelray_dict']
        # print(cur_dict.keys())
        rays, skel, directions = cur_dict['skelrays'], cur_dict['skel'], cur_dict['directions']
        mask = cur_dict['surf_mask']

        if unpack_bits:
            mask = np.unpackbits(mask)
            # print(rays.shape, skel.shape, directions.shape, mask.shape)
            mask = mask[:len(rays.reshape(-1))]
            mask = mask.reshape(rays.shape)
            mask = mask == 1

        orig_skel_size = len(skel)

        if return_reg_data:
            reg_rays = cur_dict['reg_rays']
            reg_ray_dists = cur_dict['reg_ray_dists']
            reg_skel_inds = cur_dict['reg_skel_inds']

        if transform is not None:
            # print('Transforming skelrays')
            cur_cloud, skel, directions, rays, reg_ray_dists, reg_rays = transform(cur_cloud, skel, directions, rays,
                                                                                   reg_ray_dists, reg_rays)

        if subsample_skels:

            if orig_skel_size >= num_skel_samples:
                if use_fps_sampling:
                    sample, inds_skel = fps_from_cloud(torch.FloatTensor(skel).to(fps_device), N=num_skel_samples)
                    inds_skel = inds_skel[0].detach().cpu().numpy()
                else:
                    inds_skel = np.random.choice(range(orig_skel_size), size=num_skel_samples,
                                                 replace=False)
            else:
                inds1 = list(range(orig_skel_size))
                inds2 = list(np.random.randint(0, orig_skel_size, size=num_skel_samples - orig_skel_size))
                inds_skel = np.array(inds1 + inds2)

            rays = rays[inds_skel]
            skel = skel[inds_skel]
            mask = mask[inds_skel]

            if jitter_skel_supervision:
                max_disp = torch.FloatTensor(rays.max(axis=1))
                all_max_disps += [max_disp.unsqueeze(0)]

            if return_reg_data:
                reg_skel_inds, inc_inds = remap_inds(reg_skel_inds, inds_skel,
                                                     orig_skel_size, reg_inds_sample=num_reg_sample)
                reg_rays = reg_rays[inc_inds]
                reg_ray_dists = reg_ray_dists[inc_inds]

        if return_reg_data:
            all_reg_ray_dists += [torch.FloatTensor(reg_ray_dists).unsqueeze(0)]
            all_reg_rays += [torch.FloatTensor(reg_rays).unsqueeze(0)]
            all_reg_skel_inds += [torch.LongTensor(reg_skel_inds).unsqueeze(0)]

        if sample_proportional_to_disp:
            sampling_weights = torch.FloatTensor(rays) ** 1.5
            sampling_weights[~mask] = sampling_weights[~mask] * non_surface_weight
        else:
            sampling_weights = torch.FloatTensor(mask * 1)
            sampling_weights[sampling_weights == 0] = non_surface_weight
        directions_tile = np.tile(directions[np.newaxis, :, :], (len(rays), 1, 1))
        directions_flat = directions_tile.reshape(-1, 3)
        rays_flat = rays.reshape(-1, 1)
        sampling_weights_flat = sampling_weights.reshape(-1, 1)
        inds_flat = torch.multinomial(sampling_weights_flat.t(), num_ray_queries, replacement=True)
        full_skel_ids = torch.repeat_interleave(torch.arange(len(skel)), len(directions))
        skel_ids = full_skel_ids[inds_flat]
        directions_sample = directions_flat[inds_flat]
        ray_sample = rays_flat[inds_flat]

        all_rays += [torch.FloatTensor(directions_sample)]
        all_scales += [torch.FloatTensor(ray_sample)]
        all_skel_ids += [skel_ids]
        all_clouds += [cur_cloud]
        all_skeletons += [torch.FloatTensor(skel).unsqueeze(0)]
        if return_surface_queries:
            all_surface_queries += [cur_surface_queries]

    all_clouds = torch.cat(all_clouds, axis=0).type(torch.float32)
    all_skeletons = torch.cat(all_skeletons, axis=0).type(torch.float32)
    all_skel_ids = torch.cat(all_skel_ids, axis=0).type(torch.LongTensor)
    all_rays = torch.cat(all_rays, axis=0).type(torch.float32)
    all_scales = torch.cat(all_scales, axis=0).type(torch.float32)

    if return_reg_data:
        all_reg_ray_dists = torch.cat(all_reg_ray_dists, axis=0).type(torch.float32)
        all_reg_rays = torch.cat(all_reg_rays, axis=0).type(torch.float32)
        all_reg_skel_inds = torch.cat(all_reg_skel_inds, axis=0).type(torch.LongTensor)

    if jitter_skel_supervision:
        all_max_disps = torch.cat(all_max_disps, axis=0)
        # print(all_max_disps.max(), all_max_disps.min())
        jittered_results = jitter_skeleton_supervision(all_skeletons, all_rays,
                                                       all_scales, all_skel_ids, all_reg_ray_dists,
                                                       all_reg_rays, all_reg_skel_inds,
                                                       jitter_scale=jitter_scale,
                                                       max_disps=all_max_disps)
        all_skeletons, all_rays, all_scales, all_skel_ids, all_reg_ray_dists, all_reg_rays, all_reg_skel_inds = jittered_results

    if return_reg_data:
        result = [all_clouds, all_skeletons, all_rays, all_scales, all_skel_ids, all_reg_ray_dists, all_reg_rays,
                  all_reg_skel_inds]
    else:
        result = [all_clouds, all_skeletons, all_rays, all_scales, all_skel_ids]
    if return_surface_queries:
        all_surface_queries = torch.cat(all_surface_queries, axis=0).type(torch.float32)
        result += [all_surface_queries]

    return result


def collate_reprs_finetuning(data, num_surface_sample=2048,
                               num_skel_samples=512,
                               num_queries_sample=1000,
                             init_skel_size=4096,
                             return_surface_queries=False,
                             return_occupancies=False):

        all_clouds = []
        all_skeletons = []
        all_queries = []
        all_labels = []

        for i, cur_data in enumerate(data):

            full_cloud = cur_data['cloud']
            #print(full_cloud.shape)
            inds_cloud = np.random.randint(0, full_cloud.shape[0], size=num_surface_sample)
            cur_cloud = full_cloud[inds_cloud].unsqueeze(0)

            cur_skel = cur_data['skeleton'][0].type(torch.float32)


            if return_surface_queries:
                inds_queries = np.random.randint(0, full_cloud.shape[0], size=num_queries_sample)
                cur_queries = full_cloud[inds_queries].type(torch.float32)
                cur_labels = torch.ones(len(cur_queries))
            else:
                cur_queries = cur_data['points'].type(torch.float32)
                cur_labels = cur_data['labels']
                inds_queries = np.random.randint(0, len(cur_queries), size=num_queries_sample)
                cur_queries = cur_queries[inds_queries]
                cur_labels = cur_labels[inds_queries]*1

            if len(cur_skel) < init_skel_size:
                inds1 = list(range(len(cur_skel)))
                inds2 = list(np.random.randint(0, len(cur_skel), size=init_skel_size - len(cur_skel)))
                inds_skel = np.array(inds1 + inds2)
                cur_skel = cur_skel[inds_skel]


            all_clouds += [cur_cloud]
            all_skeletons += [torch.FloatTensor(cur_skel).unsqueeze(0)]
            all_queries += [torch.FloatTensor(cur_queries).unsqueeze(0)]
            all_labels += [torch.FloatTensor(cur_labels).unsqueeze(0)]

        all_clouds = torch.cat(all_clouds, axis=0).type(torch.float32)
        all_skeletons = torch.cat(all_skeletons, axis=0).type(torch.float32)
        all_queries = torch.cat(all_queries, axis=0).type(torch.float32)
        all_labels = torch.cat(all_labels, axis=0).type(torch.float32)

        all_skeletons, _ = fps_from_cloud(all_skeletons.to('cuda'), N=num_skel_samples)

        result = [all_clouds, all_skeletons.to('cpu'), all_queries, all_labels]

        return result


def collate_reprs_combined(data, num_surface_sample=2048,
                             num_skel_samples=512,
                             num_queries_sample=1000,
                             init_skel_size=4096):
    all_clouds = []
    all_skeletons = []
    all_queries = []
    all_labels = []

    for i, cur_data in enumerate(data):

        full_cloud = cur_data['cloud']
        inds_cloud = np.random.randint(0, full_cloud.shape[0], size=num_surface_sample)
        cur_cloud = full_cloud[inds_cloud].unsqueeze(0)

        cur_skel = cur_data['skeleton'][0].type(torch.float32)

        inds_queries = np.random.randint(0, full_cloud.shape[0], size=num_queries_sample//2)
        surface_queries = full_cloud[inds_queries].type(torch.float32)
        surface_labels = -1*torch.ones(len(surface_queries))

        vol_queries = cur_data['points'].type(torch.float32)
        vol_labels = cur_data['labels']
        vol_inds_queries = np.random.randint(0, len(vol_queries), size=num_queries_sample//2)
        vol_queries = vol_queries[vol_inds_queries]
        vol_labels = vol_labels[vol_inds_queries]

        cur_queries = torch.cat((surface_queries, vol_queries), axis=0)
        cur_labels = torch.cat((surface_labels, vol_labels), axis=0)

        if len(cur_skel) < init_skel_size:
            inds1 = list(range(len(cur_skel)))
            # print(inds1)
            # print(num_skel_samples, len(cur_skel))
            inds2 = list(np.random.randint(0, len(cur_skel), size=init_skel_size - len(cur_skel)))
            inds_skel = np.array(inds1 + inds2)
            # print(inds_skel)
            cur_skel = cur_skel[inds_skel]

        all_clouds += [cur_cloud]
        all_skeletons += [torch.FloatTensor(cur_skel).unsqueeze(0)]
        all_queries += [torch.FloatTensor(cur_queries).unsqueeze(0)]
        all_labels += [torch.FloatTensor(cur_labels).unsqueeze(0)]

    all_clouds = torch.cat(all_clouds, axis=0).type(torch.float32)
    all_skeletons = torch.cat(all_skeletons, axis=0).type(torch.float32)
    all_queries = torch.cat(all_queries, axis=0).type(torch.float32)
    all_labels = torch.cat(all_labels, axis=0).type(torch.float32)

    all_skeletons, _ = fps_from_cloud(all_skeletons.to('cuda'), N=num_skel_samples)

    result = [all_clouds, all_skeletons.to('cpu'), all_queries, all_labels]

    return result


def collate_reprs_simple(data, num_surface_sample=2048,
                             num_skel_samples=512,
                             init_skel_size=4096):

    all_clouds = []
    all_skeletons = []

    for i, cur_data in enumerate(data):

        full_cloud = cur_data['cloud']
        inds_cloud = np.random.randint(0, full_cloud.shape[0], size=num_surface_sample)
        cur_cloud = full_cloud[inds_cloud].unsqueeze(0)

        cur_skel = cur_data['skeleton'][0].type(torch.float32)

        if len(cur_skel) < init_skel_size:
            inds1 = list(range(len(cur_skel)))
            inds2 = list(np.random.randint(0, len(cur_skel), size=init_skel_size - len(cur_skel)))
            inds_skel = np.array(inds1 + inds2)
            cur_skel = cur_skel[inds_skel]

        all_clouds += [cur_cloud]
        all_skeletons += [torch.FloatTensor(cur_skel).unsqueeze(0)]

    all_clouds = torch.cat(all_clouds, axis=0).type(torch.float32)
    all_skeletons = torch.cat(all_skeletons, axis=0).type(torch.float32)

    all_skeletons, _ = fps_from_cloud(all_skeletons.to('cuda'), N=num_skel_samples)

    result = [all_clouds, all_skeletons.to('cpu')]

    return result


def remap_inds(reg_inds, inds_skel, max_size, downsample_reg_inds=True, reg_inds_sample=1000):
    indicators = np.array([False] * max_size)
    indicators[inds_skel] = True
    ind_dict = dict(zip(range(max_size), indicators))
    flat_inds = reg_inds.reshape(-1)
    inc_mask = np.vectorize(ind_dict.get)(flat_inds)
    inc_mask = np.array(inc_mask).reshape(reg_inds.shape)
    inc_inds = inc_mask.sum(axis=1) == 2
    remap_values = np.array(range(len(inds_skel)))
    remap_keys = inds_skel

    remap_dict = defaultdict(list)
    for k, v in zip(remap_keys, remap_values):
        remap_dict[k] = v

    inc_inds = np.where(inc_inds)[0]
    if downsample_reg_inds:
        down_inds = np.random.randint(0, len(inc_inds), size=reg_inds_sample)
        inc_inds = inc_inds[down_inds]
    reg_inds_down = reg_inds[inc_inds]

    inds_remaped = np.vectorize(remap_dict.get)(reg_inds_down.reshape(-1))
    inds_remaped = np.array(inds_remaped).reshape(reg_inds_down.shape)

    return inds_remaped, inc_inds

