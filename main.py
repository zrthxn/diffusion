#!python

#%%
import logging
import coloredlogs
from sys import argv

from config import defaults, makeconfig, print_help
from data import ImageDataset, FacesDataset, CarsDataset
from noise import NoiseScheduler
from model import DenoisingDiffusion

#%%
ds = FacesDataset()
dl = ds.loader(defaults.batch_size)

#%%
ns = NoiseScheduler(
    ntype=defaults.noise, 
    steps=defaults.timesteps, 
    start=defaults.start,
    end=defaults.end)

for t in range(defaults.timesteps):
    im, n = ns.forward_diffusion(ds[0], t)

# %%

def train():
    pass

def test():
    pass

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    coloredlogs.install(level=logging.INFO, logger=logging.getLogger())

    # Rudimentary argument parser for command line arguments.
    # Lets us have otherwise complicated behaviour, like chaining commands.
    actions = list()
    params = list()
    for arg in argv:
        if arg.startswith("--"):
            params.append(arg[2:])
        else:
            actions.append(arg)

    if "help" in actions:
        print_help()
        exit(0)

    makeconfig(params)
    for command in actions:
        if command == "train":
            train()
        elif command == "test":
            train()
        else:
            print("Unknown command")

    exit(0)