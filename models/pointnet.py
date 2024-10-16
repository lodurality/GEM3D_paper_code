import torch
import torch.nn as nn
import torch.nn.functional as F


def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class SimplePointnet(nn.Module):
    ''' PointNet-based encoder network.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128):
        super().__init__()
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(dim, 2 * hidden_dim)
        self.fc_0 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_2 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, p):
        batch_size, T, D = p.size()

        # output size: B x T X F
        net = self.fc_pos(p)
        net = self.fc_0(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_1(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_2(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_3(self.actvn(net))

        # Recude to  B x F
        net = self.pool(net, dim=1)

        c = self.fc_c(self.actvn(net))

        return c


class ResnetPointnet(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, z_dim=128, dim=3, hidden_dim=128, vae=False):
        super().__init__()
        self.z_dim = z_dim
        self.vae = vae

        self.fc_pos = nn.Linear(dim, 2 * hidden_dim)
        self.block_0 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_1 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_2 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_3 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_4 = ResnetBlockFC(2 * hidden_dim, hidden_dim)

        if self.vae:
            self.mean_z = nn.Linear(hidden_dim, z_dim)
            self.logvar_z = nn.Linear(hidden_dim, z_dim)
        else:
            self.fc_z = nn.Linear(hidden_dim, z_dim)

        self.actvn = nn.LeakyReLU(negative_slope=0.2)
        self.pool = maxpool

    def forward(self, p, return_embs=False):
        #batch_size, T, D = p.size()
        #print(p.shape)
        # print(p.device)
        # output size: B x T X F
        net = self.fc_pos(p)
        net = self.block_0(net)
        #print(net.shape)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_1(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_2(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_3(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_4(net)

        # Recude to  B x F
        # print('encoder')
        # print(net.shape)
        if return_embs:
            return net

        ##print(net.shape)
        net = self.pool(net, dim=1)
        # print(net.shape)

        return net


class PatchResnetPointnet(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, z_dim=128, dim=3, hidden_dim=128, vae=False):
        super().__init__()
        self.z_dim = z_dim
        self.vae = vae

        self.fc_pos = nn.Linear(dim, 2 * hidden_dim)
        self.block_0 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_1 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_2 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_3 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_4 = ResnetBlockFC(2 * hidden_dim, hidden_dim)

        if self.vae:
            self.mean_z = nn.Linear(hidden_dim, z_dim)
            self.logvar_z = nn.Linear(hidden_dim, z_dim)
        else:
            self.fc_z = nn.Linear(hidden_dim, z_dim)

        self.actvn = nn.LeakyReLU(negative_slope=0.2)
        self.pool = maxpool

    def forward(self, p, return_embs=False):
        #batch_size, T, D = p.size()
        #print(p.shape)
        # print(p.device)
        # output size: B x T X F
        net = self.fc_pos(p)
        net = self.block_0(net)
        #print(net.shape)
        pooled = self.pool(net, dim=2, keepdim=True).expand(net.size())
        #print(pooled.shape, net.shape)
        net = torch.cat([net, pooled], dim=3)

        #print(net.shape)
        net = self.block_1(net)
        #print(net.shape)
        pooled = self.pool(net, dim=2, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=3)

        net = self.block_2(net)
        pooled = self.pool(net, dim=2, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=3)

        net = self.block_3(net)
        pooled = self.pool(net, dim=2, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=3)

        #print(net.shape)
        net = self.block_4(net)
        #print(net.shape)

        # Recude to  B x F
        # print('encoder')
        # print(net.shape)
        if return_embs:
            return net

        #print(net.shape)
        net = self.pool(net, dim=2)
        #print(net.shape)
        # print(net.shape)

        return net


if __name__ == "__main__":

    input = torch.randn(1,512,32,3)
    model = PatchResnetPointnet()

    out = model(input)