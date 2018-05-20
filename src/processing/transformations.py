import os
import torch

import numpy as np

from skimage import io, transform
from torchvision import transforms, utils

class ToTensor(object):
    def __call__(self, x):
        return torch.from_numpy(x).float()

class Rescale(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, img):
        h, w = img.shape ## this line might be problematic
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(img, (1, new_h, new_w))

        return img
