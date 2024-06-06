# P2PNet encoder using Point Transformer (Zhao et. al, 2020)
# Not used in the final model, provided as reference

import os.path as osp

import torch
from torch.nn import Linear as Lin
from torch_cluster import fps, knn_graph

from torch_geometric.nn import MLP, knn_interpolate
from torch_geometric.nn import MLP, PointTransformerConv, global_mean_pool, knn
from torch_geometric.utils import scatter

class TransformerBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin_in = Lin(in_channels, in_channels)
        self.lin_out = Lin(out_channels, out_channels)

        self.pos_nn = MLP([3, 64, out_channels], norm=None, plain_last=False)

        self.attn_nn = MLP([out_channels, 64, out_channels], norm=None,
                           plain_last=False)

        self.transformer = PointTransformerConv(in_channels, out_channels,
                                                pos_nn=self.pos_nn,
                                                attn_nn=self.attn_nn)

    def forward(self, x, pos, edge_index):
        x = self.lin_in(x).relu()
        x = self.transformer(x, pos, edge_index)
        x = self.lin_out(x).relu()
        return x


class TransitionDown(torch.nn.Module):
    '''
        Samples the input point cloud by a ratio percentage to reduce
        cardinality and uses an mlp to augment features dimensionnality
    '''
    def __init__(self, in_channels, out_channels, ratio=0.25, k=16):
        super().__init__()
        self.k = k
        self.ratio = ratio
        self.mlp = MLP([in_channels, out_channels], plain_last=False)

    def forward(self, x, pos, batch):
        # FPS sampling
        id_clusters = fps(pos, ratio=self.ratio, batch=batch)

        # compute for each cluster the k nearest points
        sub_batch = batch[id_clusters] if batch is not None else None

        # beware of self loop
        id_k_neighbor = knn(pos, pos[id_clusters], k=self.k, batch_x=batch,
                            batch_y=sub_batch)

        # transformation of features through a simple MLP
        x = self.mlp(x)

        # Max pool onto each cluster the features from knn in points
        x_out = scatter(x[id_k_neighbor[1]], id_k_neighbor[0], dim=0,
                        dim_size=id_clusters.size(0), reduce='max')

        # keep only the clusters and their max-pooled features
        sub_pos, out = pos[id_clusters], x_out
        return out, sub_pos, sub_batch


class TransitionUp(torch.nn.Module):
    '''
        Reduce features dimensionnality and interpolate back to higher
        resolution and cardinality
    '''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp_sub = MLP([in_channels, out_channels], plain_last=False)
        self.mlp = MLP([out_channels, out_channels], plain_last=False)

    def forward(self, x, x_sub, pos, pos_sub, batch=None, batch_sub=None):
        # transform low-res features and reduce the number of features
        x_sub = self.mlp_sub(x_sub)

        # interpolate low-res feats to high-res points
        x_interpolated = knn_interpolate(x_sub, pos_sub, pos, k=3,
                                         batch_x=batch_sub, batch_y=batch)

        x = self.mlp(x) + x_interpolated

        return x


class P2PNetPointTransformer(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=3,
                 num_disps=1,
                 dim_model=[128, 256, 512],
                 k=32,#32
                 range_max=1.0,
                 noise_length=0,
                 ):
        super().__init__()
        self.k = k
        self.num_disps = num_disps

        # dummy feature is created if there is none given
        in_channels = max(in_channels, 1)

        # first block
        self.mlp_input = MLP([in_channels, dim_model[0]], plain_last=False)

        self.transformer_input = TransformerBlock(
            in_channels=dim_model[0],
            out_channels=dim_model[0],
        )

        # backbone layers
        self.transformers_up = torch.nn.ModuleList()
        self.transformers_down = torch.nn.ModuleList()
        self.transition_up = torch.nn.ModuleList()
        self.transition_down = torch.nn.ModuleList()

        self.conv1 = torch.nn.Conv1d(dim_model[0] + noise_length, 32, 1)
        self.conv2 = torch.nn.Conv1d(32, 32, 1)
        self.conv3 = torch.nn.Conv1d(32, out_channels*num_disps, 1)
        self.ln1 = torch.nn.LayerNorm(32)
        self.ln2 = torch.nn.LayerNorm(32)
        self.range_max = range_max
        self.noise_length = noise_length

        for i in range(0, len(dim_model) - 1):

            # Add Transition Down block followed by a Point Transformer block
            self.transition_down.append(
                TransitionDown(in_channels=dim_model[i],
                               out_channels=dim_model[i + 1], k=self.k))

            self.transformers_down.append(
                TransformerBlock(in_channels=dim_model[i + 1],
                                 out_channels=dim_model[i + 1]))

            # Add Transition Up block followed by Point Transformer block
            self.transition_up.append(
                TransitionUp(in_channels=dim_model[i + 1],
                             out_channels=dim_model[i]))

            self.transformers_up.append(
                TransformerBlock(in_channels=dim_model[i],
                                 out_channels=dim_model[i]))

        # summit layers
        self.mlp_summit = MLP([dim_model[-1], dim_model[-1]], norm=None,
                              plain_last=False)

        self.transformer_summit = TransformerBlock(
            in_channels=dim_model[-1],
            out_channels=dim_model[-1],
        )

        # class score computation
        #self.mlp_output = MLP([dim_model[0], 64, out_channels], norm=None)

    def forward(self, cloud, noise=None):

        # add dummy features in case there is none
        x, pos = cloud.clone().reshape(-1, cloud.shape[-1]), cloud.clone().reshape(-1, cloud.shape[-1])
        #print(x.shape, pos.shape)
        batch = [torch.LongTensor([i] * cloud.shape[1]) for i in range(len(cloud))]
        batch = torch.cat(batch).to(x.device)
        #print(batch.shape)
        out_x = []
        out_pos = []
        out_batch = []

        # first block
        x = self.mlp_input(x)
        edge_index = knn_graph(pos, k=self.k, batch=batch)
        x = self.transformer_input(x, pos, edge_index)

        # save outputs for skipping connections
        out_x.append(x)
        out_pos.append(pos)
        out_batch.append(batch)

        # backbone down : #reduce cardinality and augment dimensionnality
        for i in range(len(self.transformers_down)):
            x, pos, batch = self.transition_down[i](x, pos, batch=batch)
            edge_index = knn_graph(pos, k=self.k, batch=batch)
            x = self.transformers_down[i](x, pos, edge_index)

            out_x.append(x)
            out_pos.append(pos)
            out_batch.append(batch)

        # summit
        #print('summit')
        #print(x.shape)
        x = self.mlp_summit(x)
        edge_index = knn_graph(pos, k=self.k, batch=batch)
        x = self.transformer_summit(x, pos, edge_index)

        # backbone up : augment cardinality and reduce dimensionnality
        n = len(self.transformers_down)
        for i in range(n):
            x = self.transition_up[-i - 1](x=out_x[-i - 2], x_sub=x,
                                           pos=out_pos[-i - 2],
                                           pos_sub=out_pos[-i - 1],
                                           batch_sub=out_batch[-i - 1],
                                           batch=out_batch[-i - 2])

            edge_index = knn_graph(out_pos[-i - 2], k=self.k,
                                   batch=out_batch[-i - 2])
            x = self.transformers_up[-i - 1](x, out_pos[-i - 2], edge_index)

        # Class score
        #print(x.shape)

        x = x.reshape(cloud.shape[0], -1, x.shape[1])
        x = x.permute(0, 2, 1)
        # print(x.shape)
        if noise is not None:
            x = torch.cat([x, noise], axis=1)

        # print(x.shape)
        #print(x.shape)
        x = self.ln1(self.conv1(x).permute(0, 2, 1)).permute(0, 2, 1)
        x = self.ln2(self.conv2(x).permute(0, 2, 1)).permute(0, 2, 1)
        x = self.conv3(x)


        x = x.permute(0, 2, 1)
        displacements = torch.sigmoid(x) * self.range_max * 2.0 - self.range_max

        return displacements

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    test_net = P2PNetPointTransformer(num_disps=4).to(device)
    test_input = torch.randn(2,2048,3).to(device)
    test_output = test_net(test_input)
    print(test_output.shape)