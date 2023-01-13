#!python
import torch
import logging
import coloredlogs
import pickle
from sys import argv
from time import time
from tqdm import tqdm
from typing import Tuple
from logging import info
from torch.nn import functional as F

from data import ImageDataset, FacesDataset, CarsDataset
from src.config import defaults, makeconfig, print_help
from src.noise import NoiseScheduler
from src.model import DenoisingDiffusion


def generate(model: torch.nn.Module, ns: NoiseScheduler):
    image = torch.randn((1, 3, 64, 64), device=defaults.device)

    for t in range(ns.steps)[::-1]:
        beta = ns.schedule[t]
        alphas_ = ns.sqrt_alphas_[t]
        alphas_rp = ns.sqrt_alpha_rp[t]
        
        # Call model (noise - prediction)
        pred = (beta * model(image, torch.tensor([t])) / alphas_)
        generated = alphas_rp * (image - pred)
        
        if t > 0:
            noise = torch.randn_like(image)
            generated += torch.sqrt(ns.posterior_variance[t]) * noise 
    
    return generated.squeeze()


def train() -> Tuple[torch.nn.Module, NoiseScheduler]:
    info("Start Training")

    if defaults.dataset == "faces":
        ds = FacesDataset()
    elif defaults.dataset == "cars":
        ds = CarsDataset()
    else:
        raise ValueError(f"Unknown dataset `{defaults.dataset}`")

    dl = ds.loader(defaults.batch_size)
    ns = NoiseScheduler(
        ntype=defaults.schedule, 
        steps=defaults.timesteps, 
        start=defaults.start,
        end=defaults.end)

    # Build model
    model = DenoisingDiffusion(defaults.shape)
    param_size = sum([p.numel() for p in model.parameters()])
    info(f"DenoisingDiffusion Model :: {param_size} parameters")

    if defaults.debug:
        print(repr(model))

    model.to(defaults.device)
    optim = torch.optim.Adam(model.parameters(), lr=defaults.lr)

    __start = time()
    for E in range(defaults.epochs):
        print(f"Epoch {E}/{defaults.epochs}")
        ImageDataset.plot([ generate(model, ns) for _ in range(8) ], 
            save=f"results/training/epoch_{E}.png")

        for batch in tqdm(dl):
            optim.zero_grad()

            timestep = torch.randint(0, ns.steps, (1,)).long()
            image, noise = ns.forward_diffusion(batch, timestep)
            noise_ = model(image, timestep)
            
            loss = F.l1_loss(noise_, noise)

            loss.backward()
            optim.step()

            if defaults.dryrun: 
                break

    __end = time()
    info(f"Training time {round((__end - __start)/(1e3 * 60), 3)} minutes.")

    return model, ns


def test(model: torch.nn.Module, ns: NoiseScheduler):
    info("Start Testing")
    ImageDataset.plot([ generate(model, ns) for _ in range(16) ], 
        save=f"results/generated.png")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    coloredlogs.install(level=logging.INFO, logger=logging.getLogger())

    # Rudimentary argument parser for command line arguments.
    # Lets us have otherwise complicated behaviour, like chaining commands.
    actions = list()
    params = list()
    for arg in argv[1:]:
        if arg == "help":
            print_help()
        elif arg.startswith("--"):
            params.append(arg[2:])
        else:
            actions.append(arg)

    # Build default config params
    makeconfig(params)

    model = None
    ns = None
    for command in actions:
        if command == "train":
            model, ns = train()
            with open("results/model.pt", "wb") as f:
                torch.save(model, f)
            with open("results/scheduler.pt", "wb") as f:
                pickle.dump(ns, f)
        
        elif command == "test":
            if model is None:
                with open("results/model.pt", "rb") as f:
                    model = torch.load(f)
            if ns is None:
                with open("results/scheduler.pt", "rb") as f:
                    ns = pickle.load(f)
            test(model, ns)
        
        else:
            print(f"Unknown command `{command}`")
            print_help()
    exit(0)