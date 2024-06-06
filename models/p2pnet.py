# adapted from https://github.com/kangxue/P2P-NET
# currently not used; provided as a reference if someone want to use it

import os.path as osp

import torch
import torch.nn.functional as F
from torch_scatter import scatter
from .pointnet import ResnetPointnet

import torch_geometric.transforms as T
try:
    from torch_geometric.nn import MLP, PointConv, fps, global_max_pool, radius, knn_interpolate
except:
    from torch_geometric.nn import MLP, PointNetConv, fps, global_max_pool, radius, knn_interpolate

class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn, use_geodesic_ball=False):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn, add_self_loops=False)
        self.use_geodesic_ball=use_geodesic_ball

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]

        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn, ):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class FPModule(torch.nn.Module):
    def __init__(self, k, nn):
        super().__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)

        return x, pos_skip, batch_skip


class P2PNetPointnet2(torch.nn.Module):
    def __init__(self, range_max=1.0,
                 noise_length=32,
                 norm='layer_norm'):
        super().__init__()

        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.0625, 0.1, MLP([3 + 3, 64, 64, 128], norm=norm))
        self.sa2_module = SAModule(64 / 384, 0.2, MLP([128 + 3, 128, 128, 256], norm=norm))
        self.sa3_module = SAModule(64 / 128, 0.4, MLP([256 + 3, 256, 256, 512], norm=norm))
        self.sa4_module = GlobalSAModule(MLP([512 + 3, 256, 512, 1024], norm=norm))

        self.fp4_module = FPModule(1, MLP([1024 + 512, 512, 512], norm=norm))
        self.fp3_module = FPModule(1, MLP([512 + 256, 512, 256], norm=norm))
        self.fp2_module = FPModule(1, MLP([256 + 128, 256, 128], norm=norm))
        self.fp1_module = FPModule(3, MLP([128 + 3, 128, 128, 128], norm=norm))

        self.conv1 = torch.nn.Conv1d(128 + noise_length, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 3, 1)
        self.ln1 = torch.nn.LayerNorm(128)
        self.ln2 = torch.nn.LayerNorm(64)
        self.range_max = range_max
        self.noise_length = noise_length

    def forward(self, cloud, noise=None):
        x, pos = cloud.clone().reshape(-1, cloud.shape[-1]), cloud.clone().reshape(-1, cloud.shape[-1])

        batch = [torch.LongTensor([i] * cloud.shape[1]) for i in range(len(cloud))]
        batch = torch.cat(batch).to(x.device)
        # print(batch.shape, x.shape, pos.shape)

        sa0_out = (x, pos, batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        sa4_out = self.sa4_module(*sa3_out)

        fp4_out = self.fp4_module(*sa4_out, *sa3_out)
        fp3_out = self.fp3_module(*fp4_out, *sa2_out)
        fp2_out = self.fp2_module(*fp3_out, *sa1_out)
        x, _, _ = self.fp1_module(*fp2_out, *sa0_out)

        # print(x.shape)
        x = x.reshape(cloud.shape[0], -1, x.shape[1])
        x = x.permute(0, 2, 1)
        # print(x.shape)
        if noise is not None:
            x = torch.cat([x, noise], axis=1)

        # print(x.shape)

        x = self.ln1(self.conv1(x).permute(0, 2, 1)).permute(0, 2, 1)
        x = self.ln2(self.conv2(x).permute(0, 2, 1)).permute(0, 2, 1)
        x = self.conv3(x)

        # x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.conv3(x)

        x = x.permute(0, 2, 1)
        # print(embs.shape)
        # print(embs)
        # print(embs.type())
        # print(self.range_max)
        # print(torch.sigmoid(embs).type())
        displacements = torch.sigmoid(x) * self.range_max * 2.0 - self.range_max

        return displacements


class P2PNetPointnet2Large(torch.nn.Module):
    def __init__(self, range_max=1.0,
                 noise_length=32,
                 norm='layer_norm'):
        super().__init__()

        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.0625, 0.1, MLP([3 + 3, 128, 128, 256], norm=norm))
        self.sa2_module = SAModule(64 / 384, 0.2, MLP([256 + 3, 256, 256, 512], norm=norm))
        self.sa3_module = SAModule(64 / 128, 0.4, MLP([512 + 3, 512, 512, 1024], norm=norm))
        self.sa4_module = GlobalSAModule(MLP([1024 + 3, 512, 1024, 2048], norm=norm))

        self.fp4_module = FPModule(1, MLP([2048 + 1024, 1024, 1024], norm=norm))
        self.fp3_module = FPModule(1, MLP([1024 + 512, 1024, 512], norm=norm))
        self.fp2_module = FPModule(1, MLP([512 + 256, 512, 256], norm=norm))
        self.fp1_module = FPModule(3, MLP([256 + 3, 256, 256, 256], norm=norm))

        self.conv1 = torch.nn.Conv1d(256 + noise_length, 256, 1)
        self.conv2 = torch.nn.Conv1d(256, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 3, 1)
        self.ln1 = torch.nn.LayerNorm(256)
        self.ln2 = torch.nn.LayerNorm(128)
        self.range_max = range_max
        self.noise_length = noise_length

    def forward(self, cloud, noise=None):
        x, pos = cloud.clone().reshape(-1, cloud.shape[-1]), cloud.clone().reshape(-1, cloud.shape[-1])

        batch = [torch.LongTensor([i] * cloud.shape[1]) for i in range(len(cloud))]
        batch = torch.cat(batch).to(x.device)
        # print(batch.shape, x.shape, pos.shape)

        sa0_out = (x, pos, batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        sa4_out = self.sa4_module(*sa3_out)

        fp4_out = self.fp4_module(*sa4_out, *sa3_out)
        fp3_out = self.fp3_module(*fp4_out, *sa2_out)
        fp2_out = self.fp2_module(*fp3_out, *sa1_out)
        x, _, _ = self.fp1_module(*fp2_out, *sa0_out)

        # print(x.shape)
        x = x.reshape(cloud.shape[0], -1, x.shape[1])
        x = x.permute(0, 2, 1)
        # print(x.shape)
        if noise is not None:
            x = torch.cat([x, noise], axis=1)

        # print(x.shape)

        x = self.ln1(self.conv1(x).permute(0, 2, 1)).permute(0, 2, 1)
        x = self.ln2(self.conv2(x).permute(0, 2, 1)).permute(0, 2, 1)
        x = self.conv3(x)

        # x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.conv3(x)

        x = x.permute(0, 2, 1)
        # print(embs.shape)
        # print(embs)
        # print(embs.type())
        # print(self.range_max)
        # print(torch.sigmoid(embs).type())
        displacements = torch.sigmoid(x) * self.range_max * 2.0 - self.range_max

        return displacements
