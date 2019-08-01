"""Data set tool of MVTEC

author: Haixin wang
e-mail: haixinwa@gmail.com
"""
import os
import re
import torch
import numpy as np
import torch.utils.data as data
from collections import OrderedDict
from .augment import *
from .eval_func import *


class Preproc(object):
    """Pre-procession of input image includes resize, crop & data augmentation

    Arguments:
        resize: tup(int width, int height): resize shape
        crop: tup(int width, int height): crop shape
    """
    def __init__(self, resize):
        self.resize = resize
        # self.crop_size = crop_size

    def __call__(self, image):
        image = cv2.resize(image, self.resize)
        # image = crop(image, crop_size=self.crop_size)
        p = np.random.uniform(0, 1)
        if (p > 0.33) and (p <= 0.66):
            image = mirror(image)
        elif p > 0.66:
            image = flip(image)

        # image normal
        image = image.astype(np.float32) / 255.
        # normalize_(tile, self.mean, self.std)
        image = torch.from_numpy(image)

        return image.unsqueeze(0)


class CHIP(data.Dataset):
    """A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection

    Arguments:
        root (string): root directory to mvtec folder.
        set (string): image set to use ('train', or 'test')
        preproc(callable, optional): pre-procession on the input image
    """

    def __init__(self, root, set, preproc=None):
        self.root = root
        self.preproc = preproc
        self.set = set

        if set == 'train':
            self.ids = list()
            set_path = os.path.join(self.root, set)
            for img in os.listdir(set_path):
                item_path = os.path.join(set_path, img)
                self.ids.append(item_path)
        elif set == 'test':
            self.test_len = 0
            self.test_dict = OrderedDict()
            set_path = os.path.join(self.root, set)
            for type in os.listdir(set_path):
                type_dir = os.path.join(set_path, type)
                if os.path.isfile(type_dir):
                    continue
                ids = list()
                for img in os.listdir(type_dir):
                    if re.search('.png', img) is None:
                        continue
                    ids.append(os.path.join(type_dir, img))
                    self.test_len += 1
                self.test_dict[type] = ids
        else:
            raise Exception("Invalid set name")

    def __getitem__(self, index):
        """Returns training image
        """
        img_path = self.ids[index]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if self.preproc is not None:
            img = self.preproc(img)

        return img

    def __len__(self):
        if self.set == 'train':
            return len(self.ids)
        else:
            return self.test_len