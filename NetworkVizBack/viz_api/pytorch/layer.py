import torch
import torch.nn as nn

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viz_api import layer
from viz_api.tensor import DataType
from viz_api.pytorch.tensor import Tensor_Torch
from viz_api.pytorch.tensor import DataType_Torch_Mapping

"""
layer is the abstraction for the underlying platform API.
As for layer, it is composed of a nn.module, input source,
output source and the parent block.

If no parent block, it will link with the upper model. We can add transform for
the input defined in the layer level. There is only one input and output source.

The input tensor is the corresponding tensor wrapper in this directory.
"""

class ReLU_Torch(layer.ReLU):
    # we can pad the linked block name to the name of layer
    def __init__(self, name: str="ReLU_Torch", inplace: bool =False):
        super(ReLU_Torch, self).__init__(name)
        self.relu = nn.ReLU(inplace=inplace)

    def get_layer(self):
        return self.relu

    def forward(self, input_tensor: Tensor_Torch):
        return Tensor_Torch(self.relu(input_tensor.get_linked_tensor()), name=self.name + "_output")

    # return KB in memory usage for feature (weight, bias), relu is 0
    def get_feature_memory_size(self):
        return sum([parameter.element_size() * parameter.nelement() for parameter in self.relu.parameters()]) / 1024

    # return KB in memory usage for gradients
    def get_grad_memory_size(self):
        return sum([0 if parameter.grad is None else parameter.grad.element_size() * parameter.grad.nelement()
                    for parameter in self.relu.parameters()]) / 1024

    def set_as_eval(self):
        self.relu.eval()

    def set_as_training(self):
        self.relu.train()

    def change_data_type(self, new_type: DataType):
        self.relu.to(DataType_Torch_Mapping[new_type])

    def set_device(self, device: torch.device):
        self.relu.to(device=device)


class Linear_Torch(layer.Linear):
    # we can pad the linked block name to the name of layer
    def __init__(self, in_features: int, out_features: int, name: str = "Linear_Torch", bias: bool = True):
        super(Linear_Torch, self).__init__(name)
        self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)

    def get_layer(self):
        return self.linear

    def forward(self, input_tensor: Tensor_Torch):
        return Tensor_Torch(self.linear(input_tensor.get_linked_tensor()), name=self.name + "_output")

    # return KB in memory usage for feature (weight, bias)
    def get_feature_memory_size(self):
        return sum([parameter.element_size() * parameter.nelement() for parameter in self.linear.parameters()]) / 1024

    # return KB in memory usage for gradients
    def get_grad_memory_size(self):
        return sum([0 if parameter.grad is None else parameter.grad.element_size() * parameter.grad.nelement()
                    for parameter in self.linear.parameters()]) / 1024

    def set_as_eval(self):
        self.linear.eval()

    def set_as_training(self):
        self.linear.train()

    def change_data_type(self, new_type: DataType):
        self.linear.to(DataType_Torch_Mapping[new_type])

    def set_device(self, device: torch.device):
        self.linear.to(device=device)


class Conv2d_Torch(layer.Conv2d):
    # we can pad the linked block name to the name of layer
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, name: str="Conv2d_Torch", stride: tuple=(1, 1), padding: tuple=(0, 0), dilation: tuple=(1, 1), groups: int=1, bias: bool=True, padding_mode: str='zeros'):
        super(Conv2d_Torch, self).__init__(name)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)

    def get_layer(self):
        return self.conv2d

    def forward(self, input_tensor: Tensor_Torch):
        return Tensor_Torch(self.conv2d(input_tensor.get_linked_tensor()), name=self.name + "_output")

    # return KB in memory usage for feature (weight, bias)
    def get_feature_memory_size(self):
        return sum([parameter.element_size() * parameter.nelement() for parameter in self.conv2d.parameters()]) / 1024

    # return KB in memory usage for gradients
    def get_grad_memory_size(self):
        return sum([0 if parameter.grad is None else parameter.grad.element_size() * parameter.grad.nelement()
                    for parameter in self.conv2d.parameters()]) / 1024

    def set_as_eval(self):
        self.conv2d.eval()

    def set_as_training(self):
        self.conv2d.eval()

    def change_data_type(self, new_type: DataType):
        self.conv2d.to(DataType_Torch_Mapping[new_type])

    def set_device(self, device: torch.device):
        self.conv2d.to(device=device)


class MaxPool2d_Torch(layer.MaxPool2d):
    # we can pad the linked block name to the name of layer
    def __init__(self, kernel_size: tuple, name: str = "MaxPool2d_Torch", stride: tuple=None, padding: tuple=(0, 0),
                 dilation: tuple=(1, 1), return_indices=False, ceil_mode=False):
        super(MaxPool2d_Torch, self).__init__(name)
        self.maxpool2d = nn.MaxPool2d(kernel_size, stride=kernel_size if None else stride, padding=padding,
                                      dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode)

    def get_layer(self):
        return self.maxpool2d

    def forward(self, input_tensor: Tensor_Torch):
        return Tensor_Torch(self.maxpool2d(input_tensor.get_linked_tensor()), name=self.name + "_output")

    # return KB in memory usage for feature (weight, bias)
    def get_feature_memory_size(self):
        return sum([parameter.element_size() * parameter.nelement() for parameter in self.maxpool2d.parameters()]) / 1024

    # return KB in memory usage for gradients
    def get_grad_memory_size(self):
        return sum([0 if parameter.grad is None else parameter.grad.element_size() * parameter.grad.nelement()
                    for parameter in self.maxpool2d.parameters()]) / 1024

    def set_as_eval(self):
        self.maxpool2d.eval()

    def set_as_training(self):
        self.maxpool2d.train()

    def change_data_type(self, new_type: DataType):
        self.maxpool2d.to(DataType_Torch_Mapping[new_type])

    def set_device(self, device: torch.device):
        self.maxpool2d.to(device=device)


class BatchNorm2d_Torch(layer.BatchNorm2d):
    # we can pad the linked block name to the name of layer,  C from an expected input of size (N, C, H, W)
    def __init__(self, num_features: int, name: str = "BatchNorm2d_Torch", eps: float=1e-05, momentum: float=0.1,
                 affine: bool=True, track_running_stats: bool=True):
        super(BatchNorm2d_Torch, self).__init__(name)
        self.batchnorm2d = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)

    def get_layer(self):
        return self.batchnorm2d

    def forward(self, input_tensor: Tensor_Torch):
        return Tensor_Torch(self.batchnorm2d(input_tensor.get_linked_tensor()), name=self.name + "_output")

    # return KB in memory usage for feature (weight, bias)
    def get_feature_memory_size(self):
        return sum([parameter.element_size() * parameter.nelement() for parameter in self.batchnorm2d.parameters()]) / 1024

    # return KB in memory usage for gradients
    def get_grad_memory_size(self):
        return sum([0 if parameter.grad is None else parameter.grad.element_size() * parameter.grad.nelement()
                    for parameter in self.batchnorm2d.parameters()]) / 1024

    def set_as_eval(self):
        self.batchnorm2d.eval()

    def set_as_training(self):
        self.batchnorm2d.train()

    def change_data_type(self, new_type: DataType):
        self.batchnorm2d.to(DataType_Torch_Mapping[new_type])

    def set_device(self, device: torch.device):
        self.batchnorm2d.to(device=device)


class MSELoss_Torch(layer.MSELoss):
    # we can pad the linked block name to the name of layer,  reduction = 'mean' | 'none' | 'sum', need exception handling
    def __init__(self, name: str = "MSELoss_Torch", reduction: str="mean"):
        super(MSELoss_Torch, self).__init__(name)
        self.mseloss = nn.MSELoss(reduction=reduction)

    def get_layer(self):
        return self.mseloss

    def forward(self, input_tensor: Tensor_Torch):
        return Tensor_Torch(self.mseloss(input_tensor.get_linked_tensor()), name=self.name + "_output")

    # return KB in memory usage for feature (weight, bias)
    def get_feature_memory_size(self):
        return sum([parameter.element_size() * parameter.nelement() for parameter in self.mseloss.parameters()]) / 1024

    # return KB in memory usage for gradients
    def get_grad_memory_size(self):
        return sum([0 if parameter.grad is None else parameter.grad.element_size() * parameter.grad.nelement()
                    for parameter in self.mseloss.parameters()]) / 1024

    def set_as_eval(self):
        self.mseloss.eval()

    def set_as_training(self):
        self.mseloss.train()

    def change_data_type(self, new_type: DataType):
        self.mseloss.to(DataType_Torch_Mapping[new_type])

    def set_device(self, device: torch.device):
        self.mseloss.to(device=device)

"""
device = torch.device("cuda:0")

input = Tensor_Torch(torch.randn(128, 20))
n = Linear_Torch(20, 30)
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    input.set_device(device=device)
    n.set_device(device)
output = n.forward(input)
print(input.name, "---", input.get_self_memory_size(), "---", input.get_grad_memory_size())
print(n.name, "---", n.get_feature_memory_size(), "---", n.get_grad_memory_size())
print(output.name, "---", output.get_self_memory_size(), "---", output.get_grad_memory_size())

import inspect
signature = inspect.signature(ReLU_Torch.__init__)
for param in signature.parameters.values():
    print(type(param.name), type(param.default), param.annotation)
"""