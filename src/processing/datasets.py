import os
import sys
import glob
import time
import torch
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from os.path import join
from os import listdir
from scipy.misc import imread, imresize, imsave
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from utils import read_textfile_by_line, mkdir, mv, cp

DATA_ENTRY = 'Data_Entry_2017.csv'
TRAIN_VAL_LIST = 'train_val_list.txt'
TRAIN_LABELS = 'train_labels.csv'

TEST_LIST = 'test_list.txt'
TEST_LABELS = 'test_labels.csv'

ENTRY_HEADERS = ['Image Index', 'Finding Labels']
DATA_DIR = './data'
IMG_DIR = join(DATA_DIR, 'images')
LABELS_DIR = join(DATA_DIR, 'labels')
TRAIN_DIR = join(DATA_DIR, 'train')
TEST_DIR = join(DATA_DIR, 'test')

CLASS_MAP = {
    'cardiomegaly': 0,
    'atelectasis': 1,
    'consolidation': 2,
    'edema': 3,
    'effusion': 4,
    'emphysema': 5,
    'fibrosis': 6,
    'hernia': 7,
    'infiltration': 8,
    'mass': 9,
    'nodule': 10,
    'pleural_thickening': 11,
    'pneumonia': 12,
    'pneumothorax': 13,
    'No Finding': 14}

def __init():
    mkdir(DATA_DIR)
    mkdir(LABELS_DIR)
    mkdir(TEST_DIR)
    mkdir(TRAIN_DIR)
    for c in CLASS_MAP.keys():
        mkdir(join(TRAIN_DIR, c))
        mkdir(join(TEST_DIR, c))
    return

def initialize_data():
    __init()
    return

def join_filename_and_labels(data_path):
    train_img_filenames = read_textfile_by_line(
        join(data_path, TRAIN_VAL_LIST))
    test_img_filenames = read_textfile_by_line(
        join(data_path, TEST_LIST))
    entry_df = pd.read_csv(
        join(data_path, DATA_ENTRY),
        usecols=ENTRY_HEADERS)

    train_df = entry_df[entry_df[ENTRY_HEADERS[0]].
        isin(train_img_filenames)]
    test_df = entry_df[entry_df[ENTRY_HEADERS[0]].
        isin(test_img_filenames)]

    # print len(train_df)
    train_df.to_csv(join(LABELS_DIR, TRAIN_LABELS))
    test_df.to_csv(join(LABELS_DIR, TEST_LABELS))

    return

def process_img(filename, labels, mode='train'):
    src = join(IMG_DIR, filename)
    dst_dir = TRAIN_DIR if mode == 'train' else TEST_DIR
    dst = [join(join(dst_dir, label), filename) for label in labels]
    for d in dst:
        cp(src, d)
    return

def partition_images(data_path):
    train_df = pd.read_csv(join(LABELS_DIR, TRAIN_LABELS))
    for index, row in train_df.iterrows():
        process_img(
            row[ENTRY_HEADERS][0],
            row[ENTRY_HEADERS][1].lower().split('|'),
            mode='train')

    test_df = pd.read_csv(join(LABELS_DIR, TEST_LABELS))
    for index, row in test_df.iterrows():
        process_img(
            row[ENTRY_HEADERS][0],
            row[ENTRY_HEADERS][1].lower().split('|'),
            mode='test')
    return

def train_val_split_loader(dataset, batch_size, num_workers=0):
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

    trainloader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    validationloader = DataLoader(dataset, batch_size=batch_size, sampler=validation_sampler, num_workers=num_workers)
    return trainloader, validationloader

def test_loader(dataset, batch_size, num_workers=0):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers)

class XrayDataset(Dataset):

    def __init__(self, c1, c2, mode='train', transform=None):
        self.c1 = join(TRAIN_DIR, c1) if mode == 'train' else join(TEST_DIR, c1)
        self.c2 = join(TRAIN_DIR, c2) if mode == 'train' else join(TEST_DIR, c2)
        self.files = [(join(self.c1, f), 0) for f in listdir(self.c1) if f.endswith('.png')]
        self.files.extend([(join(self.c2, f), 1) for f in listdir(self.c2) if f.endswith('.png')])
        self.transform = transform
        return

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename, label = self.files[idx]
        img = imread(filename, mode='L')

        if self.transform:
            img = self.transform(img)

        return (img, label)
