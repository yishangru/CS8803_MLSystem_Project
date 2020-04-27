from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image

import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image

import os
import copy

def save_vgg19_model():
    cnn = models.vgg19(pretrained=True).features

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            name = 'vgg19_conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'vgg19_relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'vgg19_pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'vgg19_bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
        i += 1
        torch.save(layer, "./vgg19/" + name + ".pt")

save_vgg19_model()
