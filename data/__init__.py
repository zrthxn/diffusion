from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch import Tensor

class ImageDataset(Dataset):
    images = list()

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx) -> Tensor:
        return self.images[idx]

    @staticmethod
    def plot(images: list, res = 4, denorm = True):
        ncols = min(len(images), 8)
        nrows = len(images) // ncols
        if ncols * nrows < len(images):
            nrows += 1

        _, ax = plt.subplots(nrows, ncols, figsize=(res * ncols, res * nrows))
        ax = ax.flatten()
        for img, ax in zip(images, ax):
            if denorm:
                img = (img + 1.) / 2.
            ax.imshow(img)
        plt.show()

    def head(self):
        self.plot(self.images[:5])

    def tail(self):
        self.plot(self.images[-5:])

    def loader(self, batch_size: int) -> DataLoader:
        return DataLoader(self, batch_size, shuffle=True)

from .faces import FacesDataset
from .cars import CarsDataset