import os
import sys
import time
import torch
import warnings
warnings.filterwarnings('ignore')

import numpy as np

from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, utils

def train_val_split(dataset, batch_size):
    ## define our indices
    num_train = len(dataset)
    indices = list(range(num_train))
    split = 4

    ## random, non-contiguous split
    validation_idx = np.random.choice(indices, size=split, replace=False)
    train_idx = list(set(indices) - set(validation_idx))

    ## define our samplers -- we use a SubsetRandomSampler because it will return
    ## a random subset of the split defined by the given indices without replacement

    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)

    trainloader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    validationloader = DataLoader(dataset, batch_size=batch_size, sampler=validation_sampler)
    return trainloader, validationloader

class ToTensor(object):
    def __call__(self, x):
        return torch.from_numpy(x).float()


class XrayTrainDataset(Dataset):
    def __init__(self, datapath):
        pass

    def __len__(self):
        pass

    def __getitem__(self):
        pass

class XrayTestDataset(Dataset):
    def __init__(self, datapath):
        pass

    def __len__(self):
        pass

    def __getitem__(self):
        pass
