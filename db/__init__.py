import cv2
import numpy as np
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


class Transform(object):
    def __init__(self, resize):
        self.resize = resize

    def __call__(self, image):
        image = cv2.resize(image, self.resize)
        cv2.imwrite('./ori.jpg', image)
        tile1 = image[0:128, 0:128]
        tile2 = image[0:128, 128:256]
        tile3 = image[128:256, 0:128]
        tile4 = image[128:256, 128:256]
        tile = np.array([tile1, tile2, tile3, tile4])
        tile = tile.astype(np.float32) / 255.
        tile = tile.transpose((0, 3, 1, 2))
        tile = torch.from_numpy(tile)

        return tile

