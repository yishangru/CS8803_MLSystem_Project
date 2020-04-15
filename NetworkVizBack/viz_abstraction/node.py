from enum import Enum, auto

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from platform.layer import layerType
import platform.pytorch.layer as torchLayer

# upper function call for corresponding layer defined in platform directory
layerTorchMapping = {
    layerType.ReLU: torchLayer.ReLU_Torch,
    layerType.Linear: torchLayer.Linear_Torch,
    layerType.Conv2d: torchLayer.Conv2d_Torch,
    layerType.MaxPool2d: torchLayer.MaxPool2d_Torch,
    layerType.BatchNorm2d: torchLayer.BatchNorm2d_Torch,
    layerType.MSELoss: torchLayer.MSELoss
}

