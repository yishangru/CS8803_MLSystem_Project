from enum import Enum, auto

class DataType(Enum):
    INT8 = auto()
    INT16 = auto()
    INT32 = auto()
    INT64 = auto()
    UINT8 = auto()
    FLOAT = auto()
    DOUBLE = auto()
    BOOLEAN = auto()


class TensorWrapper(object):
    def __init__(self, name: str):
        super(TensorWrapper, self).__init__()
        self.name = name

    # for tensor update
    def set_linked_tensor(self, linked_tensor):
        raise NotImplementedError

    def get_linked_tensor(self):
        raise NotImplementedError

    def get_view(self):
        raise NotImplementedError

    def get_data_type(self):
        raise NotImplementedError

    def get_deep_copy(self):
        raise NotImplementedError

    # return KB in memory usage for tensor
    def get_self_memory_size(self):
        raise NotImplementedError

    # return KB in memory usage for gradients
    def get_grad_memory_size(self):
        return NotImplementedError

    def remove_from_tracking_gradient(self):
        raise NotImplementedError

    def start_tracking_gradient(self):
        raise NotImplementedError

    def change_data_type(self, new_type: DataType):
        raise NotImplementedError

    def set_device(self, device):
        raise NotImplementedError

    def get_device(self):
        raise NotImplementedError