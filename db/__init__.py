import torch
from .mvtec import MVTEC
from .mvtec import Preproc as MVTEC_pre


def training_collate(batch):
    """Custom collate fn for dealing with batches of images.

    Arguments:
        batch: (tuple) A tuple of tensor images

    Return:
        (tensor) batch of images stacked on their 0 dim
    """
    # imgs = list()
    # for img in batch:
    #     _c, _h, _w = img.shape
    #     imgs.append(img.view(1, _c, _h, _w))

    return torch.stack(batch, 0)
