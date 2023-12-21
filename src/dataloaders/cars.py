from os import path, listdir
from torchvision.io import read_image

from . import ImageDataset

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
