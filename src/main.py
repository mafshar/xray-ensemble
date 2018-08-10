#!/usr/bin/env python

import os
import sys
import glob
import time
import torch
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch.nn as nn
import torch.optim as optim

from os.path import join
from processing import datasets, transformations
from processing.datasets import DATA_DIR
from deep_models.models import MODEL_DIR
from deep_models import models
from torchvision import transforms, utils

IMG_SIZE = 256

input_partitioning_flag = False

def train(train_loader, val_loader, model_path, num_epochs=10):
    net = models.ConvNet()
    weights = torch.randn(2)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, nesterov=True)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):

            inputs, labels = data

            ## zero the parameter gradients
            optimizer.zero_grad()

            ## forward + backprop + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics:
            ## print after every 4 batches
            running_loss += loss.item()
            if i % 4 == 3:
                print '==> epoch:\n\t%d' % (epoch + 1)
                print '==> samples processed:\n\t%5d' % (i + 1)
                print '==> loss:\n\t%.3f' % (running_loss / 4)
                print '-' * 30
                running_loss = 0.0

    return


def evaluate(test_loader, model_path):
    pass


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

    train_loader, val_loader = datasets.train_val_split_loader(
        train_dataset,
        batch_size=4,
        num_workers=8)

    test_loader = datasets.test_loader(
        test_dataset,
        batch_size=4,
        num_workers=8)

    models.initialize_models()

    train(
        train_loader,
        val_loader,
        join(MODEL_DIR, 'hernia_vs_no_findings'))
