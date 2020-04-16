from platform.layer import layerType
import platform.pytorch.layer as torchLayer

# upper function call for corresponding layer defined in platform directory

"""
Layer nodes can have multiple outputs. (stateless)
Constant nodes can have multiple outputs (stateless)
Transform nodes can have multiple outputs. (stateless)
Input nodes can only have one outputs (can be stateful)
"""

# for a node
class Node(object):
    def __init__(self, ):
        super(Node, self).__init__()


# layer
layerTorchMapping = {
    layerType.ReLU: torchLayer.ReLU_Torch,
    layerType.Linear: torchLayer.Linear_Torch,
    layerType.Conv2d: torchLayer.Conv2d_Torch,
    layerType.MaxPool2d: torchLayer.MaxPool2d_Torch,
    layerType.BatchNorm2d: torchLayer.BatchNorm2d_Torch,
    layerType.MSELoss: torchLayer.MSELoss
}

# input

# transform
