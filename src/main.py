#!/usr/bin/env python

import os
import sys
import glob
import time
import tqdm
import torch
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch.nn as nn
import torch.optim as optim

from os.path import join
from torchvision import transforms, utils
from tqdm import tqdm

from processing import datasets, transformations
from processing.datasets import DATA_DIR
from deep_models.models import MODEL_DIR
from deep_models import models


IMG_SIZE = 256

input_partitioning_flag = False

def train(train_loader, val_loader, model_path, lr=0.001, num_epochs=10):
    net = models.ConvNet()
    weights = torch.randn(2)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, nesterov=True)
    # begin train
    for epoch in tqdm(range(num_epochs)):
        train_loss, train_error = 0., 0.
        val_loss, val_error = 0., 0.
        correct, count, total = 0., 0., 0.
        # begin training
        for i, data in enumerate(train_loader, 0):

            inputs, labels = data

            ## zero the parameter gradients
            optimizer.zero_grad()

            ## forward + backprop + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss = loss.item()
            _, predicted = torch.max(outputs.data, 1)
            count += 1
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        # end training
        train_loss /= count
        train_error = 1 - (correct / total)
        correct, count, total = 0., 0., 0.
        # begin validation
        for i, data in enumerate(val_loader, 0):
            inputs, labels = data

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            count += 1
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        # end validation
        val_loss /= count
        val_error = 1 - (correct / total)
        # stats
        output_string = 'general:\n\tepoch --> {} | lr --> {:04.3f}\n'.format(
            epoch,
            lr
        )
        output_string += 'train:\n\tloss --> {:04.3f} | error --> {:04.3f}\n'.format(
            train_loss,
            train_error
        )
        output_string += 'validation:\n\tloss --> {:04.3f} | error --> {:04.3f}\n'.format(
            val_loss,
            val_error
        )
        # print output_string
        tqdm.write(output_string)
    # end train
    # save model
    torch.save(net, model_path)
    return

def evaluate(test_loader, model_path):
    correct = 0
    total = 0
    net = torch.load(model_path)
    for i, data in enumerate(test_loader, 0):
        inputs, labels = data
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += lables.size(0)
        correct += (predicted == labels).sum().item()
    print 'Accuracy: {:04.3f}%'.format(100 * (correct / total))
    return


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
        ]
    )

    train_dataset = datasets.XrayDataset(
        'hernia',
        'No Finding',
        mode='train',
        transform=composed_transforms
    )

    test_dataset = datasets.XrayDataset(
        'hernia',
        'No Finding',
        mode='test',
        transform=composed_transforms
    )

    train_loader, val_loader = datasets.train_val_split_loader(
        train_dataset,
        batch_size=4,
        num_workers=8
    )

    test_loader = datasets.test_loader(
        test_dataset,
        batch_size=4,
        num_workers=8
    )

    models.initialize_models()
    model_path = join(MODEL_DIR, 'hernia_vs_no_findings/model_ckpt.pt')

    train(
        train_loader,
        val_loader,
        model_path
    )

    # evaluate(
    #     test_loader,
    #     model_path
    # )
