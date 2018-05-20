#!/usr/bin/env python

import os
import sys
import glob
import time
import torch
import warnings
warnings.filterwarnings('ignore')

import numpy as np

from processing import datasets

DATA_DIR = './data'

input_partitioning_flag = False


if __name__ == '__main__':
    if input_partitioning_flag:
        datasets.initialize_data()
        datasets.join_filename_and_labels(data_path=DATA_DIR)
        datasets.partition_images(data_path=DATA_DIR)

    train_dataset = datasets.XrayDataset('hernia', 'No Finding', mode='train')
    print train_dataset[0]

    test_dataset = datasets.XrayDataset('hernia', 'No Finding', mode='test')
    print len(test_dataset)
