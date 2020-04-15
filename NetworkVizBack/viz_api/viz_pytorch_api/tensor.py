import torch

#import os, sys
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from viz_api.tensor import DataType
import viz_api.tensor as tensor

"""
data will the class wrapper for the tensor,
with meta data such as size for reference.
tensor is abstraction of data in platform.
E.g. it is tensor in PyTorch.
"""

DataType_Torch_Mapping = {
    DataType.INT8: torch.int8,
    DataType.INT16: torch.int16,
    DataType.INT32: torch.int32,
    DataType.INT64: torch.int64,
    DataType.UINT8: torch.uint8,
    DataType.FLOAT: torch.float32,
    DataType.DOUBLE: torch.double,
    DataType.BOOLEAN: torch.bool,
}

DataType_Torch_Reverse_Mapping = {
    torch.int8: DataType.INT8,
    torch.int16: DataType.INT16,
    torch.int32: DataType.INT32,
    torch.int64: DataType.INT64,
    torch.uint8: DataType.UINT8,
    torch.float32: DataType.FLOAT,
    torch.double: DataType.DOUBLE,
    torch.bool: DataType.BOOLEAN,
}

# data, device, dtype, whether tracking
class Tensor_Torch(tensor.TensorWrapper):
    def __init__(self, linked_tensor: torch.Tensor, name: str="tensor_Torch"):
        super(Tensor_Torch, self).__init__(name)
        self.set_linked_tensor(linked_tensor=linked_tensor)

    # for tensor update
    def set_linked_tensor(self, linked_tensor: torch.Tensor):
        self.linked_tensor = linked_tensor
        self.data_type = DataType_Torch_Reverse_Mapping[self.linked_tensor.dtype]

    def get_linked_tensor(self):
        return self.linked_tensor

    def get_view(self):
        return self.linked_tensor.size()

    def get_data_type(self):
        return self.data_type

    def get_deep_copy(self):
        return self.linked_tensor.clone()

    # return KB in memory usage for tensor
    def get_self_memory_size(self):
        return self.linked_tensor.element_size() * self.linked_tensor.nelement() / 1024

    # return KB in memory usage for gradients
    def get_grad_memory_size(self):
        related_grad = self.linked_tensor.grad
        return float(0) if related_grad is None else related_grad.element_size() * related_grad.nelement() / 1024

    def remove_from_tracking_gradient(self):
        self.linked_tensor.detach_()

    def start_tracking_gradient(self):
        self.linked_tensor.requires_grad_(requires_grad=True)

    def change_data_type(self, new_type: DataType):
        self.linked_tensor.to(DataType_Torch_Mapping[new_type])
        self.data_type = new_type

    def set_device(self, device: torch.device):
        if self.get_device().type != device.type:
            # self.linked_tensor.cude() will return a copy of tensor
            self.linked_tensor = self.linked_tensor.cuda(device=device)

    def get_device(self):
        return self.linked_tensor.device