"""Data set tool of MVTEC

author: Haixin wang
e-mail: haixinwa@gmail.com
"""
import os
import torch
import numpy as np
import torch.utils.data as data
from .augment import *


class Preproc(object):
    """Pre-procession of input image includes resize, crop & data augmentation

    Arguments:
        resize: tup(int width, int height): resize shape
        crop: tup(int width, int height): crop shape
    """
    def __init__(self, resize, crop_size):
        self.resize = resize
        self.crop_size = crop_size
        self.mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32)
        self.std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)

    def __call__(self, image):
        image = cv2.resize(image, self.resize)
        # tile = crop(image, crop_size=self.crop_size)
        tile = image
        p = np.random.uniform(0, 1)
        if p > 0.5:
            tile = mirror(tile)

        # image normal
        tile = tile.astype(np.float32) / 255.
        # normalize_(tile, self.mean, self.std)
        tile = tile.transpose((2, 0, 1))

        return torch.from_numpy(tile)


class MVTEC(data.Dataset):
    """A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection

    Arguments:
        root (string): root directory to mvtec folder.
        set (string): image set to use ('train', or 'test')
        preproc(callable, optional): pre-procession on the input image
    """

    def __init__(self, root, set, preproc=None):
        self.root = root
        self.preproc = preproc
        self.ids = list()

        for _item in os.listdir(root):
            item_path = os.path.join(root, _item)
            if os.path.isfile(item_path):
                continue
            if set == 'train':
                img_dir = os.path.join(item_path, set, 'good')
                for img in os.listdir(img_dir):
                    self.ids.append(os.path.join(img_dir, img))
            elif set == 'test':
                type_dir = os.path.join(item_path, set)
                for type in os.listdir(type_dir):
                    img_dir = os.path.join(item_path, set, type)
                    for img in os.listdir(img_dir):
                        self.ids.append(os.path.join(img_dir, img))
            else:
                raise Exception("Invalid set name")

    def __getitem__(self, index):
        """Returns training image
        """
        img_path = self.ids[index]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if self.preproc is not None:
            img = self.preproc(img)

        return img

    def __len__(self):
        return len(self.ids)

    def pull_image(self, index):
        """Returns test image
        """
        path = self.ids[index]
        return cv2.imread(path)

    def pull_gt(self, index):
        """Returns gt of test image
        """
        pass

# test
# if __name__ == '__main__':
#     mvtec = MVTEC(root='D:/DataSet/mvtec_anomaly_detection', set='train', preproc=None)
#     for i in range(len(mvtec)):
#         img = mvtec.__getitem__(i)
