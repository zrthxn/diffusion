import numpy as np
from typing import List
from logging import info
from os import path, listdir
from matplotlib import pyplot as plt

from torch import Tensor, no_grad
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode


class ImageDataset(Dataset):
    images = list()

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx) -> Tensor:
        return self.images[idx]

    @staticmethod
    def plot(images: list, res = 4, denorm = True, save = None):
        with no_grad():
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
                    im = np.clip(im, 0, 1)
                ax.imshow(im)
                ax.set_axis_off()
            
            fig.tight_layout(pad=2.0)
            if save is None:
                plt.show()
            else:
                plt.savefig(save)
            plt.close()

    def head(self):
        self.plot(self.images[:5])

    def tail(self):
        self.plot(self.images[-5:])

    def loader(self, batch_size: int) -> DataLoader:
        return DataLoader(self, batch_size, shuffle=True, drop_last=True)


class FacesDataset(ImageDataset):
    """
    Smiling or Not Face Data from Kaggle.
    """
    
    path = 'data/faces'

    def __init__(self, device = 'cpu', norm = True) -> None:
        info("Building Faces Dataset")
        paths = list()
        for subdir in ['smile', 'non_smile', 'test']:
            for f in listdir(path.join(self.path, subdir)):
                if f.endswith('jpg'): 
                    paths.append(path.join(self.path, subdir, f))

        for f in paths:
            try:
                im = read_image(f, mode=ImageReadMode.RGB).to(device)
                if norm:
                    im = ((im / 255.) * 2.) - 1.
                self.images.append(im)
            except Exception as e:
                print(Warning(e))
                continue

        info(f"Read {len(self.images)} face images")


class CarsDataset(ImageDataset):
    """
    Car Images Data.
    """
    
    path = 'data/cars'

    def __init__(self, norm = True) -> None:
        for f in listdir(self.path):
            if f.endswith('jpg'): 
                im = read_image(path.join(self.path, f)).permute(1, 2, 0)
                if norm:
                    im = ((im / 255.) * 2.) - 1.
                self.images.append(im)
