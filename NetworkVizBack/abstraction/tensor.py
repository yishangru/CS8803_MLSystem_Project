import os
import sys
from enum import Enum, auto

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import platform.pytorch.tensor as torch_tensor

class dataType(Enum):
    INT8 = auto()
    INT16 = auto()
    INT32 = auto()
    UINT8 = auto()
    FLOAT = auto()
    DOUBLE = auto()
    BOOLEAN = auto()

dataTypeTorchMapping = {
    dataType.INT8: torch.int8,
    dataType.INT16: torch.int16,
    INT32 = torch.int32
    UINT8 = torch.uint8
    FLOAT = torch.float32
    DOUBLE = torch.double
    BOOLEAN = torch.bool
}

class dataTypeWrapper(object):
    def __init__(self, name: str):
        super(dataTypeWrapper, self).__init__()
        self.name = name

class tensorWrapper(object):
    def __init__(self, name: str):
        super(tensorWrapper, self).__init__()
        self.name = name

    def add_linked_tensor(self):
        return NotImplementedError

    def get_linked_tensor(self):
        return NotImplementedError

    def get_view(self):
        return NotImplementedError

    def get_data_type(self):
        return NotImplementedError

    def get_true_data_type(self):
        return NotImplementedError

    def get_memory_size(self):
        return NotImplementedError

    def remove_from_tracking_gradient(self):
        pass