import torch

from .shapenet import ShapeNet, ShapeNetSkel

class AxisScaling(object):
    def __init__(self, interval=(0.75, 1.25), jitter=True):
        assert isinstance(interval, tuple)
        self.interval = interval
        self.jitter = jitter
        
    def __call__(self, surface, point):
        scaling = torch.rand(1, 3) * 0.5 + 0.75
        surface = surface * scaling
        point = point * scaling

        scale = (1 / torch.abs(surface).max().item()) * 0.999999
        surface *= scale
        point *= scale

        if self.jitter:
            surface += 0.005 * torch.randn_like(surface)
            surface.clamp_(min=-1, max=1)

        return surface, point


def build_shape_surface_occupancy_dataset(split, args):
    if split == 'train':
        # transform = #transforms.Compose([
        transform = AxisScaling((0.75, 1.25), True)
        # ])
        return ShapeNet(args.data_path, split=split, transform=transform, sampling=True, num_samples=1024, return_surface=True, surface_sampling=True, pc_size=args.point_cloud_size)
    elif split == 'val':
        # return ShapeNet(args.data_path, split=split, transform=None, sampling=True, num_samples=1024, return_surface=True, surface_sampling=True, pc_size=args.point_cloud_size)
        return ShapeNet(args.data_path, split=split, transform=None, sampling=False, return_surface=True, surface_sampling=True, pc_size=args.point_cloud_size)
    else:
        return ShapeNet(args.data_path, split=split, transform=None, sampling=False, return_surface=True, surface_sampling=True, pc_size=args.point_cloud_size)


def build_shape_surface_skeleton_dataset(split, args, dataset_object=ShapeNetSkel,
                                         return_surface=True):
    if split == 'train':
        # transform = #transforms.Compose([

        return dataset_object(args.data_path, split=split, transform=None, sampling=True,
                            num_samples=1024, return_surface=return_surface, surface_sampling=True,
                            pc_size=args.point_cloud_size,
                            data_subsample=args.data_subsample,
                            use_dilg_scale=args.use_dilg_scale,
                            occupancies_base_folder=args.occupancies_base_folder,
                            num_skel_samples=args.num_skel_samples,
                            use_fps=args.use_fps)
    elif split == 'val':
        # return ShapeNet(args.data_path, split=split, transform=None, sampling=True, num_samples=1024, return_surface=True, surface_sampling=True, pc_size=args.point_cloud_size)
        return dataset_object(args.data_path, split=split, transform=None, sampling=False,
                            return_surface=return_surface, surface_sampling=True,
                            pc_size=args.point_cloud_size, data_subsample=args.data_subsample,
                            use_dilg_scale=args.use_dilg_scale,
                            occupancies_base_folder=args.occupancies_base_folder,
                            num_skel_samples=args.num_skel_samples,
                            use_fps=args.use_fps)
    else:
        return dataset_object(args.data_path, split=split, transform=None, sampling=False,
                            return_surface=return_surface, surface_sampling=True,
                            pc_size=args.point_cloud_size, data_subsample=args.data_subsample,
                            use_dilg_scale=args.use_dilg_scale,
                            occupancies_base_folder=args.occupancies_base_folder,
                            num_skel_samples=args.num_skel_samples,
                            use_fps=args.use_fps)



if __name__ == '__main__':
    # m = ShapeNet('/home/zhanb0b/data/', 'train', transform=AxisScaling(), sampling=True, num_samples=1024, return_surface=True, surface_sampling=True)
    m = ShapeNet('/home/zhanb0b/data/', 'train', transform=AxisScaling(), sampling=True, num_samples=1024, return_surface=True, surface_sampling=True)
    p, l, s, c = m[0]
    print(p.shape, l.shape, s.shape, c)
    print(p.max(dim=0)[0], p.min(dim=0)[0])
    print(p[l==1].max(axis=0)[0], p[l==1].min(axis=0)[0])
    print(s.max(axis=0)[0], s.min(axis=0)[0])