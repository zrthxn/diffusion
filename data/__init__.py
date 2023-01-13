from logging import info
from typing import List
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch import Tensor, no_grad, clamp

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
                img = img.detach().permute(1, 2, 0)
                if denorm:
                    img = (img + 1.) / 2.
                    img = clamp(img, 0.0, 1.0)
                ax.imshow(img)
                ax.set_axis_off()
            
            fig.tight_layout(pad=2.0)
            if save is None:
                plt.show()
            else:
                plt.savefig(save)

    def head(self):
        self.plot(self.images[:5])

    def tail(self):
        self.plot(self.images[-5:])

    def loader(self, batch_size: int) -> DataLoader:
        info(f"Building DataLoader with {len(self.images) // batch_size} batches of {batch_size} samples")
        return DataLoader(self, batch_size, shuffle=True)

from .faces import FacesDataset
from .cars import CarsDataset