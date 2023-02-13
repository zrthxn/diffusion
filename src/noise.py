import json
import torch
from typing import Union
from torch import Tensor
from torch.nn import functional as F
from logging import info

linear = lambda start, end, steps: torch.linspace(start, end, steps)
cosine = lambda start, end, steps: torch.sin(linear(start, end, steps))
expont = lambda start, end, steps: torch.exp(linear(start, end, steps))


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
        elif ntype == 'cosine':
            self.schedule = cosine(start, end, steps).to(device)
        elif ntype == 'exponential':
            self.schedule = expont(start, end, steps).to(device)
        else:
            raise ValueError('Unknown noise schedule type')
        
        alphas = (1. - self.schedule).to(device)
        alphacp = torch.cumprod(alphas, axis=0).to(device)
        alphacp_shift = F.pad(alphacp[:-1], (1,0), value=1.0).to(device)

        self.sqrt_alpha_rp = torch.sqrt(1. / alphas).to(device)

        self.sqrt_alphacp = torch.sqrt(alphacp).to(device)
        self.sqrt_oneminus_alphacp = torch.sqrt(1. - alphacp).to(device)
        
        self.posterior_variance = self.schedule * (1. - alphacp_shift) / (1. - alphacp)

    def forward_diffusion(self, image: Tensor, timestep: Union[int, Tensor]) -> Tensor:
        noise = torch.rand_like(image)

        sqrt_a_t = self.sqrt_alphacp[timestep]
        sqrt_a_t_ = self.sqrt_oneminus_alphacp[timestep]

        if type(timestep) == Tensor:
            shape = torch.Size( ( image.shape[0], *((1,) * (len(image.shape) - 1)) ) )
            sqrt_a_t = sqrt_a_t.reshape(shape)
            sqrt_a_t_ = sqrt_a_t_.reshape(shape)

        diff = (sqrt_a_t * image) + (sqrt_a_t_ * noise) 
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
