import json
import torch
from torch import Tensor
from torch.nn import functional as F
from logging import info


linear = lambda steps, start, end: torch.linspace(start, end, steps)
cosine = lambda steps, start, end: torch.sin(linear(steps, start, end))
expont = lambda steps, start, end: torch.exp(linear(steps, start, end))


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

        if ntype == 'linear':
            self.schedule = linear(steps, start, end).to(device)
        elif ntype == 'cosine':
            self.schedule = cosine(steps, start, end).to(device)
        elif ntype == 'exponential':
            self.schedule = expont(steps, start, end).to(device)
        else:
            raise ValueError('Unknown noise schedule type')
        
        alphas = (1. - self.schedule).to(device)
        alphacp = torch.cumprod(alphas, axis=0).to(device)
        alphacp_shift = F.pad(alphacp[:-1], (1,0), value=1.0).to(device)

        self.sqrt_alpha_rp = torch.sqrt(1. / alphas).to(device)

        self.sqrt_alphacp = torch.sqrt(alphacp).to(device)
        self.oneminus_sqrt_alphacp = torch.sqrt(1. - alphacp).to(device)
        
        self.posterior_variance = self.schedule * (1. - alphacp_shift) / (1. - alphacp)

    def forward_diffusion(self, input_: Tensor, timestep: int) -> Tensor:
        noise = torch.rand_like(input_)

        sqrta_t = self.sqrt_alphacp[timestep]
        sqrta_t_ = self.oneminus_sqrt_alphacp[timestep]

        diff = (sqrta_t * input_) + (sqrta_t_ * noise) 
        return diff, noise
    
    @torch.no_grad()
    def sample(self, model):
        image = torch.randn((1, 3, 64, 64), device=model.device)

        for t in range(self.steps)[::-1]:
            beta = self.schedule[t]
            alphas_ = self.oneminus_sqrt_alphacp[t]
            alphas_rp = self.sqrt_alpha_rp[t]
            
            # Call model (noise - prediction)
            step = torch.tensor([t], device=model.device)
            noise_ = (beta / alphas_) * model(image, step)
            image_ = alphas_rp * (image - noise_)

            if t > 0:
                sampled_noise = torch.randn_like(image)
                image = image_ + torch.sqrt(self.posterior_variance[t]) * sampled_noise
            else:
                image = image_

        return image.squeeze()

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
