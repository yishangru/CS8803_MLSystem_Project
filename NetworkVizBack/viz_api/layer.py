from enum import Enum, auto
from viz_api.tensor import TensorWrapper

class LayerType(Enum):
    ReLU = auto()
    Linear = auto()
    Conv2d = auto()
    MaxPool2d = auto()
    BatchNorm2d = auto()
    MSELoss = auto()

class LayerWrapper(object):
    def __init__(self, name: str):
        super(LayerWrapper, self).__init__()
        self.name = name

    def forward(self, input_tensor: TensorWrapper):
        raise NotImplementedError

    def get_memory_size(self):
        return NotImplementedError


class ReLU(LayerWrapper):
    def __init__(self, name: str):
        super(ReLU, self).__init__(name)


class Conv2d(LayerWrapper):
    def __init__(self, name: str):
        super(Conv2d, self).__init__(name)


class MaxPool2d(LayerWrapper):
    def __init__(self, name: str):
        super(MaxPool2d, self).__init__(name)


class BatchNorm2d(LayerWrapper):
    def __init__(self, name: str):
        super(BatchNorm2d, self).__init__(name)


class MSELoss(LayerWrapper):
    def __init__(self, name: str):
        super(MSELoss, self).__init__(name)