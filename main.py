#!python
import os
import torch
import logging
from time import time
from tqdm import tqdm
from logging import info
from torch.nn import functional as F
from matplotlib import pyplot as plt
from upycli import command

from src.dataloaders import ImageDataset, FacesDataset, CarsDataset
from src.noise import NoiseScheduler
from src.model import DenoisingDiffusion
from src.utils import plot, write_gif

logging.basicConfig(level=logging.INFO)


@command
def train(device = "cpu",
        dataset = "faces",
        debug = False,
        dryrun = False,
        output_dir = ".",
        
        # Noise Scheduler parameters
        schedule = "linear",
        timesteps = 300,
        start = 0.0001,
        end = 0.020,
        
        # Training parameterss
        lr = 0.001,
        epochs = 50,
        batch_size = 16,
        
        # Model shape
        shape = (
            # Downsampling
            [64, 128, 256, 512, 1024],
            # Upsampling
            [1024, 512, 256, 128, 64])):
    """ Train a new model from scratch.
    """
    
    info("Start Training")
    if dataset == "faces":
        ds = FacesDataset(device)
    elif dataset == "cars":
        ds = CarsDataset()
    else:
        raise ValueError(f"Unknown dataset `{dataset}`")
    
    info(f"Built {len(ds.images) // batch_size} batches of {batch_size} samples")

    ns = NoiseScheduler(
        ntype=schedule, 
        steps=timesteps, 
        start=start,
        end=end,
        device=device)

    # Build model
    model = DenoisingDiffusion(shape)
    param_size = sum([p.numel() for p in model.parameters()])
    info(f"DenoisingDiffusion Model :: {param_size} parameters")

    if debug:
        print(repr(model))

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if not os.path.exists(f"{output_dir}/.checkpoints"):
        os.mkdir(f"{output_dir}/.checkpoints")
    ns.save(f"{output_dir}/.checkpoints/scheduler.json")

    __start = time()
    losslog = list()
    for E in range(1, epochs + 1):
        print(f"Epoch {E}/{epochs}", f"Epoch Loss {losslog[-1]}" if losslog else "")
        
        dl = ds.loader(batch_size)
        for batch in tqdm(dl):
            optimizer.zero_grad()

            timestep = torch.randint(0, ns.steps, 
                size=(batch_size,), 
                device=device,
                dtype=torch.long)
            
            image_, noise = ns.forward_diffusion(batch, timestep)
            noise_ = model(image_, timestep)
            
            loss = F.l1_loss(noise, noise_)
            loss.backward()
            optimizer.step()
            
            losslog.append(loss.detach().cpu().item())

            if dryrun: 
                break

        # Save checkpoint for this epoch
        torch.save(model, f"{output_dir}/.checkpoints/epoch_{E}.pt")

        plt.figure(figsize=(12,4), dpi=150)
        plt.semilogy(losslog)
        plt.savefig(f"{output_dir}/losslog.png")
        plt.close()
        
        plot(model.sample(ns, 8), save=f"{output_dir}/training/epoch_{E}.png")

    __end = time()
    info(f"Training time {round((__end - __start)/60, 3)} minutes.")

    torch.save(model, f"{output_dir}/model.pt")
    ns.save(f"{output_dir}/scheduler.json")


@command
def test(model: str, ns_path: str, output_dir: str, device = "cpu"):
    """ Run a model from a given path.
    """
    
    info("Start Testing")
    model = torch.load(model, map_location=device)
    ns = NoiseScheduler.load(ns_path, device=device)
    
    image, evo = model.sample(ns, 16, return_evolution=True)
    plot(model.sample(ns, 16), save=f"{output_dir}/generated.png")
    write_gif(evo, save=f"{output_dir}/evolution.gif", fps=8)
