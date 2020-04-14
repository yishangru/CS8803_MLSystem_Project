import torch
from enum import Enum, auto

class DataType(Enum):
    INT8 = auto()
    INT16 = auto()
    INT32 = auto()
    UINT8 = auto()
    FLOAT = auto()
    DOUBLE = auto()
    BOOLEAN = auto()


class TensorWrapper(object):
    def __init__(self, name: str):
        super(TensorWrapper, self).__init__()
        self.name = name

    def get_linked_tensor(self):
        raise NotImplementedError

    def get_view(self):
        raise NotImplementedError

    def get_data_type(self):
        raise NotImplementedError

    # return KB in memory usage for tensor
    def get_self_memory_size(self):
        raise NotImplementedError

    # return KB in memory usage for gradients
    def get_grad_memory_size(self):
        return NotImplementedError

    def remove_from_tracking_gradient(self):
        raise NotImplementedError

    def change_data_type(self, new_type: DataType):
        raise NotImplementedError

    def set_device(self, device: torch.device):
        raise NotImplementedError

    def get_device(self):
        raise NotImplementedError