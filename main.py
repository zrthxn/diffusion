#!python
import os
import torch
import logging
from sys import argv
from time import time
from tqdm import tqdm
from typing import Tuple
from logging import info
from torch.nn import functional as F
from matplotlib import pyplot as plt

from data import ImageDataset, FacesDataset, CarsDataset
from src.config import defaults, makeconfig, print_help
from src.noise import NoiseScheduler
from src.model import DenoisingDiffusion


def train() -> Tuple[torch.nn.Module, NoiseScheduler]:
    info("Start Training")

    if defaults.dataset == "faces":
        ds = FacesDataset()
    elif defaults.dataset == "cars":
        ds = CarsDataset()
    else:
        raise ValueError(f"Unknown dataset `{defaults.dataset}`")
    
    info(f"Built {len(ds.images) // defaults.batch_size} batches of {defaults.batch_size} samples")

    ns = NoiseScheduler(
        ntype=defaults.schedule, 
        steps=defaults.timesteps, 
        start=defaults.start,
        end=defaults.end,
        device=defaults.device)

    # Build model
    model = DenoisingDiffusion(defaults.shape)
    param_size = sum([p.numel() for p in model.parameters()])
    info(f"DenoisingDiffusion Model :: {param_size} parameters")

    if defaults.debug:
        print(repr(model))

    model.to(defaults.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=defaults.lr)

    if not os.path.exists('.checkpoints'):
        os.mkdir('.checkpoints')
    ns.save(f".checkpoints/scheduler.json")

    __start = time()
    losslog = list()
    for E in range(1, defaults.epochs + 1):
        print(f"Epoch {E}/{defaults.epochs}", f"Epoch Loss {losslog[-1]}" if losslog else "")
        
        dl = ds.loader(defaults.batch_size)
        for batch in tqdm(dl):
            optimizer.zero_grad()

            timestep = torch.randint(0, ns.steps, 
                size=(defaults.batch_size,), 
                device=defaults.device,
                dtype=torch.long)
            
            image_, noise = ns.forward_diffusion(batch, timestep)
            noise_ = model(image_, timestep)
            
            loss = F.l1_loss(noise, noise_)
            loss.backward()
            optimizer.step()
            
            losslog.append(loss.detach().cpu().item())

            if defaults.dryrun: 
                break

        # Save checkpoint for this epoch
        torch.save(model, f".checkpoints/epoch_{E}.pt")

        plt.figure(figsize=(12,4), dpi=150)
        plt.semilogy(losslog)
        plt.savefig("results/losslog.png")
        plt.close()
        
        ImageDataset.plot(model.sample(ns, 8), save=f"results/training/epoch_{E}.png")

    __end = time()
    info(f"Training time {round((__end - __start)/60, 3)} minutes.")

    return model, ns, losslog


def test(model: DenoisingDiffusion, ns: NoiseScheduler):
    info("Start Testing")

    # model = torch.load("results/model.pt", map_location="cpu")
    # ns = NoiseScheduler.load("results/scheduler.json")
    # model = model.eval()

    # with torch.no_grad():
    #     img = torch.randn((1, 3, 64, 64), device=ns.device)
    #     img = img / ( img.max() - img.min() )
    #     shape = torch.Size( ( img.shape[0], *((1,) * (len(img.shape) - 1)) ) )

    #     for i in range(ns.steps)[::-1]:
    #         t = torch.full((1,), i, device=ns.device)

    #         beta = ns.schedule[t].reshape(shape)
    #         alphas_ = ns.sqrt_oneminus_alphacp[t].reshape(shape)
    #         alphas_rp = torch.sqrt(1. / ns.alphas[t]).reshape(shape)
            
    #         # Call model (noise - prediction)
    #         mean = alphas_rp * (img - beta * model(img, t) / alphas_)

    #         if torch.isnan(mean).any():
    #             print(
    #                 "B1",
    #                 ns.schedule[i+1],
    #                 ns.sqrt_oneminus_alphacp[i+1],
    #                 torch.sqrt(1. / ns.alphas[i+1]),
    #                 ns.posterior_variance[i+1]
    #             )

    #             print(model(img, t))
    #             break
    #         else:
    #             if i > 0:
    #                 noise = torch.randn_like(img)
    #                 img = mean + torch.sqrt(ns.posterior_variance[t]).reshape(shape) * noise
    #             else:
    #                 img = mean

    #             if torch.isnan(img).any():
    #                 print(
    #                     "B2",
    #                     mean,
    #                     torch.sqrt(ns.posterior_variance[t]).reshape(shape),
    #                     noise
    #                 )
    #                 break

    #     im = img.detach().squeeze().permute(1, 2, 0).cpu().numpy()
    #     im = im / (im.max() - im.min())
    #     im = (im + 1.) / 2.
        
    #     plt.imshow(im)
    #     plt.show()
    #     plt.close()

    ImageDataset.plot(model.sample(ns, 16), save=f"results/generated.png")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

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
            model, ns, losslog = train()
            torch.save(model, "results/model.pt")
            ns.save("results/scheduler.json")
        
        elif command == "test":
            if model is None:
                model = torch.load("results/model.pt", map_location=defaults.device)
            if ns is None:
                try:
                    ns = NoiseScheduler.load("results/scheduler.json", device=defaults.device)
                except:
                    ns = NoiseScheduler(
                        ntype=defaults.schedule, 
                        steps=defaults.timesteps, 
                        start=defaults.start,
                        end=defaults.end,
                        device=defaults.device)
            test(model, ns)
        
        else:
            print(f"Unknown command `{command}`")
            print_help()
    exit(0)