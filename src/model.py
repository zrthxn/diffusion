import math
import torch
from torch import nn
from torch.nn import functional as F
from logging import info

from .noise import NoiseScheduler

class DownsampleBlock(nn.Module):
    def __init__(self, 
        in_channels: int, 
        out_channels: int, 
        time_emb_dim: int):

        super().__init__()
        
        self.time_mlp =  nn.Linear(time_emb_dim, out_channels)
        self.in_bnorm = nn.BatchNorm2d(out_channels)
        
        self.in_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.transform = nn.Conv2d(out_channels, out_channels, 4, 2, 1)
    
        self.out_bnorm = nn.BatchNorm2d(out_channels)
        self.out_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x, t, ):
        # First Conv
        h = self.in_bnorm(F.relu(self.in_conv(x)))

        # Time embedding
        time_emb = F.relu(self.time_mlp(t))
        
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        
        # Add time channel
        h = h + time_emb
        
        # Second Conv
        h = self.out_bnorm(F.relu(self.out_conv(h)))
        
        # Down or Upsample
        return self.transform(h)


class UpsampleBlock(nn.Module):
    def __init__(self, 
        in_channels: int, 
        out_channels: int, 
        time_emb_dim: int):

        super().__init__()
        
        self.time_mlp =  nn.Linear(time_emb_dim, out_channels)
        self.in_bnorm = nn.BatchNorm2d(out_channels)
        
        self.in_conv = nn.Conv2d(2*in_channels, out_channels, kernel_size=3, padding=1)
        self.transform = nn.ConvTranspose2d(out_channels, out_channels, 4, 2, 1)
        
        self.out_bnorm = nn.BatchNorm2d(out_channels)
        self.out_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x, t, ):
        # First Conv
        h = self.in_bnorm(F.relu(self.in_conv(x)))

        # Time embedding
        time_emb = F.relu(self.time_mlp(t))
        
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        
        # Add time channel
        h = h + time_emb
        
        # Second Conv
        h = self.out_bnorm(F.relu(self.out_conv(h)))
        
        # Down or Upsample
        return self.transform(h)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half_dim = self.dim // 2
        
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim) * -embeddings).to(time.device)

        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class DenoisingDiffusion(nn.Module):
    out_dim = 1 
    time_emb_dim = 32
    image_channels = 3

    def __init__(self, shape: tuple) -> None:
        super().__init__()
        info("Initialize Model")

        self.down_channels, self.up_channels = shape
        
        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(self.time_emb_dim),
                nn.Linear(self.time_emb_dim, self.time_emb_dim),
                nn.ReLU()
            )
        
        # Initial projection
        self.projection = nn.Conv2d(self.image_channels, self.down_channels[0], 3, padding=1)

        # Downsample
        self.downsample = nn.ModuleList([
            DownsampleBlock(
                self.down_channels[i], 
                self.down_channels[i+1], 
                self.time_emb_dim) \
                for i in range(len(self.down_channels)-1) ])

        # Upsample
        self.upsample = nn.ModuleList([
            UpsampleBlock(
                self.up_channels[i], 
                self.up_channels[i+1], 
                self.time_emb_dim) \
                for i in range(len(self.up_channels)-1) ])

        self.output = nn.Conv2d(self.up_channels[-1], 3, self.out_dim)

    def forward(self, x, timestep):
        # Embedd time
        t = self.time_mlp(timestep)

        # Initial conv
        x = self.projection(x)

        # Downsampling
        residuals = list()
        for down in self.downsample:
            x = down(x, t)
            residuals.append(x)

        # Upsampling
        for up in self.upsample:
            # Add residual x as additional channels
            x = torch.cat((x, residuals.pop()), dim=1)           
            x = up(x, t)
        
        return self.output(x)

    @torch.no_grad()
    def sample(self, ns: NoiseScheduler):
        image = torch.randn((1, 3, 64, 64), device=ns.device)
        shape = torch.Size( ( image.shape[0], *((1,) * (len(image.shape) - 1)) ) )

        for i in range(0, ns.steps)[::-1]:
            t = torch.full((1,), i, device=ns.device, dtype=torch.long)

            beta = ns.schedule[t].reshape(shape)
            alphas_ = ns.sqrt_oneminus_alphacp[t].reshape(shape)
            alphas_rp = torch.sqrt(1. / ns.alphas[t]).reshape(shape)

            # Call model (noise - prediction)
            mean = alphas_rp * (image - beta * self(image, t) / alphas_)

            if i > 0:
                noise = torch.randn_like(image)
                image = mean + torch.sqrt(ns.posterior_variance[t]) * noise
            else:
                image = mean

        return image.squeeze()
