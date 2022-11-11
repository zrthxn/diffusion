from os import path, listdir
from torchvision.io import read_image

from . import ImageDataset

class FacesDataset(ImageDataset):
    """
    Smiling or Not Face Data from Kaggle.
    """
    
    path = 'data/faces'

    def __init__(self, norm = True) -> None:
        paths = list()
        for subdir in ['smile', 'non_smile', 'test']:
            for f in listdir(path.join(self.path, subdir)):
                if f.endswith('jpg'): paths.append(path.join(self.path, subdir, f))

        for f in paths:
            im = read_image(f).permute(1, 2, 0)
            if norm:
                im = ((im / 255.) * 2.) - 1.
            self.images.append(im)
