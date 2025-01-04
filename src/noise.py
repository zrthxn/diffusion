import json
import torch
import math
from torch import Tensor
from torch.nn import functional as F
from logging import info


linear = lambda start, end, steps: torch.linspace(start, end, steps)
expont = lambda start, end, steps: torch.exp(linear(math.log(start), math.log(end), steps))
cosine = lambda start, end, steps: (end - start)/2 * (torch.cos(linear(torch.pi, 2 * torch.pi, steps)) + 1) + start


class NoiseScheduler:
    """
    Noise Scheduler
    """

    def __init__(self, 
        ntype: str, 
        steps: int, 
        start = 0.001, 
        end = 1.000,
        device = 'cpu') -> None:
        
        info(f"Using {ntype} noise schedule")
        info(f"Scheduled from {start} to {end} in {steps} steps")

        self.ntype = ntype
        self.steps = steps
        self.start = start
        self.end = end
        self.device = device

        if ntype == 'linear':
            self.schedule = linear(start, end, steps).to(device)
        elif ntype == 'exponential':
            self.schedule = expont(start, end, steps).to(device)
        elif ntype == 'cosine':
            self.schedule = cosine(start, end, steps).to(device)
        else:
            raise ValueError(f'Unknown noise schedule type `{ntype}`')
        
        self.alphas = (1. - self.schedule).to(device)

        alphacp = torch.cumprod(self.alphas, axis=0).to(device)
        alphacp_shift = F.pad(alphacp[:-1], (1,0), value=1.0).to(device)

        self.sqrt_alphacp = torch.sqrt(alphacp).to(device)
        self.sqrt_oneminus_alphacp = torch.sqrt(1. - alphacp).to(device)
        
        self.posterior_variance = self.schedule * (1. - alphacp_shift) / (1. - alphacp)

    def forward_diffusion(self, image: Tensor, timestep: Tensor) -> Tensor:
        noise = torch.randn_like(image, device=image.device)
        shape = torch.Size( ( image.shape[0], *((1,) * (len(image.shape) - 1)) ) )

        mean = self.sqrt_alphacp[timestep].reshape(shape)
        vari = self.sqrt_oneminus_alphacp[timestep].reshape(shape)
        
        diff = (mean * image) + (vari * noise) 
        return diff, noise

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump({
                "ntype": self.ntype, 
                "steps": self.steps, 
                "start": self.start,
                "end": self.end
            }, f)
    
    @classmethod
    def load(cls, path: str, device = 'cpu'):
        with open(path, 'r') as f:
            JSON = json.load(f)
        return cls(**JSON, device=device)
