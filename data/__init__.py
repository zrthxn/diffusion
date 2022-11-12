from logging import info
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch import Tensor, no_grad

class ImageDataset(Dataset):
    images = list()

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx) -> Tensor:
        return self.images[idx]

    @no_grad()
    @staticmethod
    def plot(images: list, res = 4, denorm = True):
        ncols = min(len(images), 8)
        nrows = len(images) // ncols
        if ncols * nrows < len(images):
            nrows += 1

        _, ax = plt.subplots(nrows, ncols, figsize=(res * ncols, res * nrows))
        ax = ax.flatten()
        for img, ax in zip(images, ax):
            img = img.detach().permute(1, 2, 0)
            if denorm:
                img = (img + 1.) / 2.
            ax.imshow(img)
        plt.show()

    def head(self):
        self.plot(self.images[:5])

    def tail(self):
        self.plot(self.images[-5:])

    def loader(self, batch_size: int) -> DataLoader:
        info(f"Building DataLoader with {len(self.images) // batch_size} batches of {batch_size} samples")
        return DataLoader(self, batch_size, shuffle=True)

from .faces import FacesDataset
from .cars import CarsDataset