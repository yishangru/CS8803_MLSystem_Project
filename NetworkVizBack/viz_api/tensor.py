from enum import Enum, auto
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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
        return NotImplementedError

    def get_view(self):
        return NotImplementedError

    def change_data_type(self, new_type: DataType):
        return NotImplementedError

    def get_data_type(self):
        return NotImplementedError

    def get_memory_size(self):
        return NotImplementedError

    def remove_from_tracking_gradient(self):
        return NotImplementedError