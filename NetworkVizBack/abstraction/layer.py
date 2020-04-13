import os
import sys
from enum import Enum, auto

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import platform.pytorch.layer as torchLayer


class layerType(Enum):
    ReLU = auto()
    Linear = auto()
    Conv2d = auto()
    MaxPool2d = auto()
    BatchNorm2d = auto()
    MSELoss = auto()


layerTorchMapping = {
    layerType.ReLU: torchLayer.ReLU_Torch,
    layerType.Linear: torchLayer.Linear_Torch,
    layerType.Conv2d: torchLayer.Conv2d_Torch,
    layerType.MaxPool2d: torchLayer.MaxPool2d_Torch,
    layerType.BatchNorm2d: torchLayer.BatchNorm2d_Torch,
    layerType.MSELoss: torchLayer.MSELoss
}


class layerWrapper(object):
    def __init__(self, name):
        super(layerWrapper, self).__init__()
        self.name = name

    def forward(self):
        raise NotImplementedError


class ReLU(layerWrapper):
    def __init__(self, name):
        super(ReLU, self).__init__(name)

    def forward(self):
        pass


class Conv2d(layerWrapper):
    def __init__(self, name):
        super(Conv2d, self).__init__(name)

    def forward(self):
        pass


class MaxPool2d(layerWrapper):
    def __init__(self, name):
        super(MaxPool2d, self).__init__(name)

    def forward(self):
        pass


class BatchNorm2d(layerWrapper):
    def __init__(self, name):
        super(BatchNorm2d, self).__init__(name)

    def forward(self):
        pass


class MSELoss(layerWrapper):
    def __init__(self, name):
        super(MSELoss, self).__init__(name)

    def forward(self):
        pass