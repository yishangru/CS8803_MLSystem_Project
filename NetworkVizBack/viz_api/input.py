import torch
from enum import Enum, auto
from viz_api.tensor import TensorWrapper
from viz_api.tensor import DataType

"""
torchvision datasets
trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
valset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)
DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None)
"""

class InputType(Enum):
    ImageLoader = auto()    # load single image
    RandomLoader = auto()   # generate random tensor with shape
    DateSetLoader = auto()  # load training set or validation set
    ConstantLoader = auto() # generate constant tensor with shape (zero or one)

# data loader, it will contain a tensor which holder the data
class InputWrapper(object):
    def __init__(self, name: str):
        super(InputWrapper, self).__init__()
        self.name = name

    def get_loaded_tensor(self):
        pass

    def set_device(self, device):
        raise NotImplementedError

    def get_device(self):
        raise NotImplementedError

    # return KB in memory usage for the loaded tensor
    def get_tensor_memory_size(self):
        raise NotImplementedError

    # return KB in memory usage for gradients of the loaded tensor
    def get_tensor_grad_memory_size(self):
        return NotImplementedError

    def get_despcription(self):
        raise NotImplementedError

# load one image
class ImageLoader(InputWrapper):
    def __init__(self, name: str):
        super(ImageLoader, self).__init__(name)

# generate random tensor with shape
class RandomLoader(InputWrapper):
    def __init__(self, name: str):
        super(RandomLoader, self).__init__(name)

# load training set or validation set
class ImageDateSetLoader(InputWrapper):
    def __init__(self, name: str):
        super(ImageDateSetLoader, self).__init__(name)

# generate constant tensor with shape (zero or one)
class ConstantLoader(InputWrapper):
    def __init__(self, name: str):
        super(ConstantLoader, self).__init__(name)