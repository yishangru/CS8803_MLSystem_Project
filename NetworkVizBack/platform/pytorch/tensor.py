import torch
from enum import Enum

"""
data will the class wrapper for the tensor,
with meta data such as size for reference.
tensor is abstraction of data in platform.
E.g. it is tensor in PyTorch.
"""
class dataType_Torch(object)

class dataType(Enum):
    INT8 = torch.int8
    INT16 = torch.int16
    INT32 = torch.int32
    UINT8 = torch.uint8
    FLOAT = torch.float32
    DOUBLE = torch.double
    BOOLEAN = torch.bool

# data, device, dtype, whether tracking
class tensor_Torch(tensorWrapper):
    def __init__(self):
        super(tensorWrapperTorch, self).__init__()

    def add_linked_tensor(self, linked_data, operations: list):
        # operations is a list of expected operation for tensor
        return self.linked_tensor

    def get_linked_tensor(self):
        return self.linked_tensor

    def get_view(self):
        return self.linked_tensor.size()

    def get_data_type(self):
        return self.dataType

    def get_true_data_type(self):
        return repr(self.dataType)

    def get_memory_size(self):
        return self.linked_tensor.element_size() * self.linked_tensor.nelement() / 1024

    def remove_from_tracking_gradient(self):
        self.linked_tensor.detach()