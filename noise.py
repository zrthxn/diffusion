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
        self.device = device

        if ntype == 'linear':
            self.schedule = linear(steps, start, end)
        elif ntype == 'cosine':
            self.schedule = cosine(steps, start, end)
        elif ntype == 'exponential':
            self.schedule = expont(steps, start, end)
        else:
            raise ValueError('Unknown noise schedule type')
        
        alphas = 1 - self.schedule
        alphcp = torch.cumprod(alphas, axis=0)
        alphcp_shift = F.pad(alphcp[:-1], (1,0), value=1.0)

        self.sqrt_alpha_rp = torch.sqrt(1. / alphas)

        self.sqrt_alphas = torch.sqrt(alphcp)
        self.sqrt_alphas_ = torch.sqrt(1 - alphcp)
        
        self.posterior_variance = self.schedule * (1. - alphcp_shift) / (1. - alphcp)

    def forward_diffusion(self, input_: Tensor, timestep: int) -> Tensor:
        noise = torch.rand_like(input_)

        sqrta_t = self.sqrt_alphas[timestep]
        sqrta_t_ = self.sqrt_alphas_[timestep]

        diff = (sqrta_t * input_) + (sqrta_t_ * noise) 
        return diff.to(self.device), noise.to(self.device)
