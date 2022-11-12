#!python

import torch
import logging
import coloredlogs
from sys import argv
from time import time
from tqdm import tqdm
from logging import info
from torch.nn import functional as F

from config import defaults, makeconfig, print_help
from data import ImageDataset, FacesDataset, CarsDataset
from noise import NoiseScheduler
from model import DenoisingDiffusion


def train(ds: ImageDataset) -> torch.nn.Module:
    info("Start Training")

    dl = ds.loader(defaults.batch_size)
    ns = NoiseScheduler(
        ntype=defaults.noise, 
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
        for batch in tqdm(dl):
            optim.zero_grad()

            timestep = torch.randint(0, ns.steps, (1,)).long()
            image, noise = ns.forward_diffusion(batch, timestep)
            noise_ = model(image, timestep)
            
            loss = F.l1_loss(noise_, noise)

            loss.backward()
            optim.step()
    __end = time()
    info(f"Training time was {(__end - __start)/(1e3 * 60)} minutes.")

    return model


def test():
    info("Start Testing")
    pass


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
    
    if defaults.dataset == "faces":
        ds = FacesDataset()
    elif defaults.dataset == "cars":
        ds = CarsDataset()
    else:
        raise ValueError(f"Unknown dataset `{defaults.dataset}`")

    for command in actions:
        if command == "train":
            train(ds)
        elif command == "test":
            test()
        elif command == "load": pass
        elif command == "plot": pass
        else:
            print(f"Unknown command `{command}`")
            print_help()
    exit(0)