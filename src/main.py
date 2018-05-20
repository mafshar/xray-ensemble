#!/usr/bin/env python

import os
import sys
import glob
import time
import torch
import warnings
warnings.filterwarnings('ignore')

import numpy as np

from processing import datasets, transformations
from torchvision import transforms, utils

DATA_DIR = './data'
IMG_SIZE = 256

input_partitioning_flag = False


if __name__ == '__main__':
    if input_partitioning_flag:
        datasets.initialize_data()
        datasets.join_filename_and_labels(data_path=DATA_DIR)
        datasets.partition_images(data_path=DATA_DIR)

    scale = transformations.Rescale(IMG_SIZE)
    tensorize = transformations.ToTensor()
    composed_transforms = transforms.Compose(
        [scale,
        tensorize,
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    train_dataset = datasets.XrayDataset(
        'hernia',
        'No Finding',
        mode='train',
        transform=composed_transforms)

    test_dataset = datasets.XrayDataset(
        'hernia',
        'No Finding',
        mode='test',
        transform=composed_transforms)

    train_loader, val_loader = datasets.train_val_split(
        train_dataset,
        batch_size=4,
        num_workers=8)

    test_loader = datasets.test_loader(
        test_dataset,
        batch_size=4,
        num_workers=8)
