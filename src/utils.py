# import numpy as np
import torch
from typing import List
from matplotlib import pyplot as plt
from torch import Tensor, no_grad


@no_grad()
def plot(images: List[torch.Tensor], res = 4, denorm = True, save = None):
    ncols = min(len(images), 8)
    nrows = len(images) // ncols
    if ncols * nrows < len(images):
        nrows += 1

    fig, ax = plt.subplots(nrows, ncols, 
        figsize=(res * ncols, res * nrows),
        sharey=True,
        sharex=True)

    fig.set_dpi(240)
    axes: List[plt.Axes] = ax.flatten()
    for img, ax in zip(images, axes):
        im = img.detach().permute(1, 2, 0).cpu().numpy()
        if denorm:
            im = (im + 1.) / 2.
            im = torch.clip(im, 0, 1)
        ax.imshow(im)
        ax.set_axis_off()
    
    fig.tight_layout(pad=2.0)
    if save is None:
        plt.show()
    else:
        plt.savefig(save)
    plt.close()


@no_grad()
def write_gif(gif: List[torch.Tensor], path: str, fps = 60):
    fig = plt.figure()
    frames = [[plt.imshow(frame.detach().cpu().numpy(), animated=True)] for frame in gif]
    
    animation.ArtistAnimation(fig, frames, 
        interval=int((1/fps)*1000), 
        blit=True, 
        repeat_delay=1000).save(path)
    
    plt.close(fig)
