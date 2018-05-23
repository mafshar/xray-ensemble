import os
import sys
import time
import torch

from os.path import join

from processing.utils import mkdir
from processing.datasets import CLASS_MAP

import torch.nn as nn
import torch.nn.functional as F

MODEL_DIR = './models'

def initialize_models():
    __init()

def __init():
    mkdir(MODEL_DIR)
    for c in CLASS_MAP.keys():
        if c != 'No Finding':
            dir = join(MODEL_DIR, c + '_vs_no_findings')
            mkdir(dir)
    return

class ConvNet(nn.Module):
    '''
    Input       --> 1x256x256 image
    ConvLayer1  --> 1 input channel, 10 output channels, 5x5 conv; ReLU Activation
    ConvLayer2  --> 10 input channels, 20 output channels, 5x5 conv; ReLU Activation
    ConvLayer3  --> 20 input channels, 40 output channels, 5x5 conv; ReLU Activation
    ConvLayer4  --> 40 input channels, 64 output channels, 5x5 conv; ReLU Activation
    ConvLayer5  --> 64 input channels, 64 output channels, 5x5 conv; ReLU Activation
    FC1         --> fully connected layer with 1024 hidden units; ReLU Activation
    Droput      --> dropout layer with p(dropout) = 0.2
    FC2         -->

    '''

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5) ## 1 input channel, 10 feature maps (kernels), 5x5 convolution
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.conv3 = nn.Conv2d(20, 40, 5)
        self.conv4 = nn.Conv2d(40, 64, 5)
        self.conv5 = nn.Conv2d(64, 64, 5)
        self.fc1 = nn.Linear(64 * 5 * 5, 1024)
        self.fc2 = nn.Linear(1024, 2)
        return

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) ## 2x2 max pooling
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv4(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv5(x)), (2, 2))
        x = x.view(-1, self.__num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def __num_flat_features(self, x):
        size = x.size()[1:] ## all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
