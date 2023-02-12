import numpy as np
from typing import List
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch import Tensor, no_grad

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

from .faces import FacesDataset
from .cars import CarsDataset