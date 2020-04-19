from enum import Enum, auto
from viz_api.tensor import DataType

class LayerType(Enum):
    ReLU = auto()
    Linear = auto()
    Conv2d = auto()
    MaxPool2d = auto()
    BatchNorm2d = auto()
    LogSoftMax = auto()
    MSELoss = auto()
    NLLLoss = auto()


class LayerWrapper(object):
    def __init__(self, name: str):
        super(LayerWrapper, self).__init__()
        self.name = name

    def get_layer(self):
        raise NotImplementedError

    def forward(self, *input):
        raise NotImplementedError

    # return KB in memory usage for feature (weight, bias)
    def get_feature_memory_size(self):
        raise NotImplementedError

    # return KB in memory usage for gradients
    def get_grad_memory_size(self):
        raise NotImplementedError

    def set_as_eval(self):
        raise NotImplementedError

    def set_as_training(self):
        raise NotImplementedError

    def change_data_type(self, new_type: DataType):
        raise NotImplementedError

    def set_device(self, device):
        raise NotImplementedError

    def get_device(self):
        raise NotImplementedError

    @staticmethod
    def get_description(self):
        raise NotImplementedError


class ReLU(LayerWrapper):
    def __init__(self, name: str):
        super(ReLU, self).__init__(name)


class Linear(LayerWrapper):
    def __init__(self, name: str):
        super(Linear, self).__init__(name)


class Conv2d(LayerWrapper):
    def __init__(self, name: str):
        super(Conv2d, self).__init__(name)


class MaxPool2d(LayerWrapper):
    def __init__(self, name: str):
        super(MaxPool2d, self).__init__(name)


class BatchNorm2d(LayerWrapper):
    def __init__(self, name: str):
        super(BatchNorm2d, self).__init__(name)


class LogSoftMax(LayerWrapper):
    def __init__(self, name: str):
        super(LogSoftMax, self).__init__(name)


class MSELoss(LayerWrapper):
    def __init__(self, name: str):
        super(MSELoss, self).__init__(name)


class NLLLoss(LayerWrapper):
    def __init__(self, name: str):
        super(NLLLoss, self).__init__(name)