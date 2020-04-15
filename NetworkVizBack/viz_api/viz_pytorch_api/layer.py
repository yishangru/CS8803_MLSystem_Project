import torch
import torch.nn as nn

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viz_api import layer
from viz_api.tensor import DataType
from viz_api.viz_pytorch_api.tensor import Tensor_Torch
from viz_api.viz_pytorch_api.tensor import DataType_Torch_Mapping

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
    def __init__(self, name: str="ReLU_Torch", import_relu: nn.ReLU=None, inplace: bool =False, device=torch.device("cpu")):
        super(ReLU_Torch, self).__init__(name)
        self.relu = nn.ReLU(inplace=inplace) # always new a ReLU
        if device.type == "cuda":
            self.relu.to(device)
        self.device = device

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
        self.device = device

    def get_device(self):
        return self.device

    def get_despcription(self):
        return "ReLu Layer"


class Linear_Torch(layer.Linear):
    # we can pad the linked block name to the name of layer
    def __init__(self, in_features: int, out_features: int, import_linear: nn.Linear = None, name: str = "Linear_Torch", bias: bool = True, device=torch.device("cpu")):
        super(Linear_Torch, self).__init__(name)
        self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=bias) if import_linear is None else import_linear
        if device.type == "cuda":
            self.linear.to(device)
        self.device = device

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
        self.device = device

    def get_device(self):
        return self.device

    def get_despcription(self):
        return "Linear Layer"


class Conv2d_Torch(layer.Conv2d):
    # we can pad the linked block name to the name of layer
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, import_conv2d: nn.Conv2d = None, name: str="Conv2d_Torch",
                 stride: tuple=(1, 1), padding: tuple=(0, 0), dilation: tuple=(1, 1), groups: int=1, bias: bool=True, padding_mode: str='zeros', device=torch.device("cpu")):
        super(Conv2d_Torch, self).__init__(name)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode) if import_conv2d is None else import_conv2d
        if device.type == "cuda":
            self.conv2d.to(device)
        self.device = device

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
        self.device = device

    def get_device(self):
        return self.device

    def get_despcription(self):
        return "Conv2d Layer"


class MaxPool2d_Torch(layer.MaxPool2d):
    # we can pad the linked block name to the name of layer
    def __init__(self, kernel_size: tuple, import_maxpool2d: nn.MaxPool2d = None, name: str = "MaxPool2d_Torch", stride: tuple=None, padding: tuple=(0, 0),
                 dilation: tuple=(1, 1), return_indices=False, ceil_mode=False, device=torch.device("cpu")):
        super(MaxPool2d_Torch, self).__init__(name)
        self.maxpool2d = nn.MaxPool2d(kernel_size, stride=kernel_size if None else stride, padding=padding,
                                      dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode) if import_maxpool2d is None else import_maxpool2d
        if device.type == "cuda":
            self.maxpool2d.to(device)
        self.device = device

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
        self.device = device

    def get_device(self):
        return self.device

    def get_despcription(self):
        return "MaxPool2d Layer"


class BatchNorm2d_Torch(layer.BatchNorm2d):
    # we can pad the linked block name to the name of layer,  C from an expected input of size (N, C, H, W)
    def __init__(self, num_features: int, import_batchnorm2d: nn.BatchNorm2d = None, name: str = "BatchNorm2d_Torch", eps: float=1e-05, momentum: float=0.1,
                 affine: bool=True, track_running_stats: bool=True, device=torch.device("cpu")):
        super(BatchNorm2d_Torch, self).__init__(name)
        self.batchnorm2d = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats) if import_batchnorm2d is None else import_batchnorm2d
        if device.type == "cuda":
            self.batchnorm2d.to(device)
        self.device = device

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
        self.device = device

    def get_device(self):
        return self.device

    def get_despcription(self):
        return "Batch2dNorm Layer"


class LogSoftmax_Torch(layer.LogSoftMax):
    # we can pad the linked block name to the name of layer
    def __init__(self, dim: int, import_logsoftmax: nn.LogSoftmax = None, name: str = "LogSoftMax_Torch",
                 device=torch.device("cpu")):
        super(LogSoftmax_Torch, self).__init__(name)
        self.logsoftmax = nn.LogSoftmax(dim=dim) if import_logsoftmax is None else import_logsoftmax
        if device.type == "cuda":
            self.logsoftmax.to(device)
        self.device = device

    def get_layer(self):
        return self.logsoftmax

    def forward(self, input_tensor: Tensor_Torch):
        return Tensor_Torch(self.logsoftmax(input_tensor.get_linked_tensor()), name=self.name + "_output")

    # return KB in memory usage for feature (weight, bias)
    def get_feature_memory_size(self):
        return sum([parameter.element_size() * parameter.nelement() for parameter in self.logsoftmax.parameters()]) / 1024

    # return KB in memory usage for gradients
    def get_grad_memory_size(self):
        return sum([0 if parameter.grad is None else parameter.grad.element_size() * parameter.grad.nelement()
                    for parameter in self.logsoftmax.parameters()]) / 1024

    def set_as_eval(self):
        self.logsoftmax.eval()

    def set_as_training(self):
        self.logsoftmax.train()

    def change_data_type(self, new_type: DataType):
        self.logsoftmax.to(DataType_Torch_Mapping[new_type])

    def set_device(self, device: torch.device):
        self.logsoftmax.to(device=device)
        self.device = device

    def get_device(self):
        return self.device

    def get_despcription(self):
        return "LogSoftMax Layer"


class MSELoss_Torch(layer.MSELoss):
    # we can pad the linked block name to the name of layer,  reduction = 'mean' | 'none' | 'sum', need exception handling
    def __init__(self, import_mseloss: nn.MSELoss = None, name: str = "MSELoss_Torch", reduction: str="mean", device=torch.device("cpu")):
        super(MSELoss_Torch, self).__init__(name)
        self.mseloss = nn.MSELoss(reduction=reduction) if import_mseloss is None else import_mseloss
        if device.type == "cuda":
            self.mseloss.to(device)
        self.device = device

    def get_layer(self):
        return self.mseloss

    def forward(self, input_tensor: Tensor_Torch, target_tensor: Tensor_Torch):
        return Tensor_Torch(self.mseloss(input_tensor.get_linked_tensor(), target_tensor.get_linked_tensor()), name=self.name + "_output")

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
        self.device = device

    def get_device(self):
        return self.device

    def get_despcription(self):
        return "MSELoss Layer"


class NLLLoss_Torch(layer.NLLLoss):
    # we can pad the linked block name to the name of layer,  reduction = 'mean' | 'none' | 'sum', need exception handling
    def __init__(self, import_nllloss: nn.MSELoss = None, name: str = "NLLLoss_Torch", reduction: str="mean", device=torch.device("cpu")):
        super(NLLLoss_Torch, self).__init__(name)
        self.nllloss = nn.NLLLoss(reduction=reduction) if import_nllloss is None else import_nllloss
        if device.type == "cuda":
            self.nllloss.to(device)
        self.device = device

    def get_layer(self):
        return self.nllloss

    def forward(self, input_tensor: Tensor_Torch, target_tensor: Tensor_Torch):
        return Tensor_Torch(self.nllloss(input_tensor.get_linked_tensor(), target_tensor.get_linked_tensor()), name=self.name + "_output")

    # return KB in memory usage for feature (weight, bias)
    def get_feature_memory_size(self):
        return sum([parameter.element_size() * parameter.nelement() for parameter in self.nllloss.parameters()]) / 1024

    # return KB in memory usage for gradients
    def get_grad_memory_size(self):
        return sum([0 if parameter.grad is None else parameter.grad.element_size() * parameter.grad.nelement()
                    for parameter in self.nllloss.parameters()]) / 1024

    def set_as_eval(self):
        self.nllloss.eval()

    def set_as_training(self):
        self.nllloss.train()

    def change_data_type(self, new_type: DataType):
        self.nllloss.to(DataType_Torch_Mapping[new_type])

    def set_device(self, device: torch.device):
        self.nllloss.to(device=device)
        self.device = device

    def get_device(self):
        return self.device

    def get_despcription(self):
        return "NLLLoss Layer"


def test_logsoftmax():
    device = torch.device("cuda:0")
    m = LogSoftmax_Torch(dim=1, device=device)
    input = Tensor_Torch(torch.randn(2, 3, device=device))
    output = m.forward(input)

    print(input.get_device(), m.get_device(), output.get_device())
    print(input.name, "---", input.get_self_memory_size(), "---", input.get_grad_memory_size())
    print(m.name, "---", m.get_feature_memory_size(), "---", m.get_grad_memory_size())
    print(output.name, "---", output.get_self_memory_size(), "---", output.get_grad_memory_size())

"""
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
input = Tensor_Torch(torch.randn(128, 20))
input.set_device(device=device)
n = Linear_Torch(20, 30, device=device)
output = n.forward(input)

print(input.get_device(), n.get_device(), output.get_device())
print(input.name, "---", input.get_self_memory_size(), "---", input.get_grad_memory_size())
print(n.name, "---", n.get_feature_memory_size(), "---", n.get_grad_memory_size())
print(output.name, "---", output.get_self_memory_size(), "---", output.get_grad_memory_size())

import inspect
signature = inspect.signature(ReLU_Torch.__init__)
for param in signature.parameters.values():
    print(type(param.name), type(param.default), param.annotation)
"""