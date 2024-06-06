from torch.nn import Sequential as Seq, Linear as Lin
import torch.nn as nn
import torch

def MLP(channels, bn=False, activation=nn.ReLU()):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), activation)
        for i in range(1, len(channels)-1)] + [Lin(channels[len(channels)-2], channels[len(channels)-1])])


class DiagonalGaussianDistribution(object):
    def __init__(self, mean, logvar, deterministic=False):
        self.mean = mean
        self.logvar = logvar
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.mean.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.mean.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.mean(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2])
            else:
                return 0.5 * torch.mean(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean



class KLAutoEncoder(nn.Module):
    def __init__(
        self,
        *,
        depth=24,
        dim=512,
        queries_dim=512,
        output_dim = 1,
        num_inputs = 2048,
        num_latents = 512,
        latent_dim = 64,
        heads = 8,
        dim_head = 64,
        weight_tie_layers = False,
        decoder_ff = False
    ):
        super().__init__()
        
        self.depth = depth

        self.num_inputs = num_inputs
        self.num_latents = num_latents

        
        #self.proj_back = MLP([latent_dim,128,256,dim]) 
        self.proj_back = nn.Linear(latent_dim, dim)

        #self.mean_fc = MLP([256,128,64,latent_dim]) 
        #self.logvar_fc = MLP([256,128,64,latent_dim])
        
        self.mean_fc = nn.Linear(dim, latent_dim)
        self.logvar_fc = nn.Linear(dim, latent_dim)

    def encode(self, pc):
        # pc: B x N x 3
        B, N, D = pc.shape
        #assert N == self.num_inputs
        
        mean = self.mean_fc(pc)
        logvar = self.logvar_fc(pc)

        #print(mean.shape, logvar.shape)

        posterior = DiagonalGaussianDistribution(mean, logvar)
        x = posterior.sample()
        kl = posterior.kl()

        return kl, x
    
    
    def encode_means(self, pc):
        # pc: B x N x 3
        B, N, D = pc.shape
        
        mean = self.mean_fc(pc)
        
        return mean


    def decode(self, x):

        x = self.proj_back(x)

        return x

    def forward(self, pc):
        kl, x_encoded = self.encode(pc)
        #print(x_encoded.shape)
        
        x_decoded = self.decode(x_encoded)
        return kl, x_decoded


class StandarScaler(nn.Module):
    def __init__(
            self,
            *,
            dim=256,
            latent_dim=256,

    ):
        super().__init__()
        assert dim == latent_dim, 'dim needs to be equal to latent_dim for standard scaler'
        self.latent_dim = latent_dim
        self.register_buffer('means', torch.zeros(self.latent_dim))
        self.register_buffer('stds', torch.ones(self.latent_dim))

    def encode(self, pc):
        # pc: B x N x 3
        B, N, D = pc.shape
        # assert N == self.num_inputs

        kl = None
        x = pc.double() - self.means.double()[None, None, :]
        x = x / (self.stds.double()[None, None, :] + 1e-6)

        return kl, x.float()

    def encode_means(self, pc):
        # pc: B x N x 3
        B, N, D = pc.shape
        # assert N == self.num_inputs

        kl = None
        print(self.means.double()[None, None, :].shape)
        x = pc.double() - self.means.double()[None, None, :]
        print(x.max())
        x = x / (self.stds.double()[None, None, :] + 1e-6)
        print(x.max())
        return x.float()

    def decode(self, x):
        B, N, D = x.shape
        # assert N == self.num_inputs

        x = x * (self.stds.double()[None, None, :] + 1e-6)
        x = x.double() + self.means.double()[None, None, :]

        return x.float()

    def forward(self, pc):
        kl, x_encoded = self.encode(pc)
        # print(x_encoded.shape)

        x_decoded = self.decode(x_encoded)
        return kl, x_decoded