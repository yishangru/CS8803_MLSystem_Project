from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim

from .information import blockInformation

"""
    For block level abstraction:
        2. Linked hardware
"""


class Block(object):
    def __init__(self, blockInfo: blockInformation):  # block information
        super(Block, self).__init__()