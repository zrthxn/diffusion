import math
import torch
from torch import nn
from torch.nn import functional as F


class DenoisingBlock(nn.Module):
    def __init__(self, 
        in_channels: int, 
        out_channels: int, 
        time_emb_dim: int, 
        upsample = False,
        image_channels = 3):

        super().__init__()
        
        self.time =  nn.Linear(time_emb_dim, out_channels)
        self.in_bnorm = nn.BatchNorm2d(out_channels)
        self.out_bnorm = nn.BatchNorm2d(out_channels)
        
        if upsample:
            self.in_conv = nn.Conv2d(2*in_channels, out_channels, image_channels, padding=1)
            self.transform = nn.ConvTranspose2d(out_channels, out_channels, 4, 2, 1)
        else:
            self.in_conv = nn.Conv2d(in_channels, out_channels, image_channels, padding=1)
            self.transform = nn.Conv2d(out_channels, out_channels, 4, 2, 1)
        
        self.out_conv = nn.Conv2d(out_channels, out_channels, image_channels, padding=1)
        
    def forward(self, x, t, ):
        # First Conv
        h = self.in_bnorm(F.relu(self.in_conv(x)))

        # Time embedding
        time_emb = F.relu(self.time(t))
        
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
        embeddings = torch.exp(torch.arange(half_dim) * -embeddings)

        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class DenoisingDiffusion(nn.Module):
    down_channels = (64, 128, 256, 512, 1024)
    up_channels = (1024, 512, 256, 128, 64)
    
    out_dim = 1 
    time_emb_dim = 32
    image_channels = 3

    def __init__(self) -> None:
        super().__init__()

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
            DenoisingBlock(self.down_channels[i], self.down_channels[i+1], self.time_emb_dim) \
                for i in range(len(self.down_channels)-1) ])

        # Upsample
        self.upsample = nn.ModuleList([
            DenoisingBlock(self.up_channels[i], self.up_channels[i+1], self.time_emb_dim, up=True) \
                for i in range(len(self.up_channels)-1) ])

        self.output = nn.Conv2d(self.up_channels[-1], 3, self.out_dim)

    def forward(self, x, timestep):
        # Embedd time
        t = self.time_mlp(timestep)

        # Initial conv
        x = self.projection(x)

        # Downsampling
        residuals = [ down(x, t) for down in self.downsample ]

        # Upsampling
        for up in self.upsample:
            # Add residual x as additional channels
            x = torch.cat((x, residuals.pop()), dim=1)           
            x = up(x, t)
        
        return self.output(x)
