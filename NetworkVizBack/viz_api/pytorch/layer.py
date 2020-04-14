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

    def forward(self, input_tensor: Tensor_Torch):
        return Tensor_Torch(self.relu(input_tensor.get_linked_tensor()), name=self.name + "_output")

    # return KB in memory usage, relu is 0 in pytorch
    def get_memory_size(self):
        return sum([parameter.element_size() * parameter.nelement() for parameter in self.relu.parameters()])

    def set_as_eval(self):
        self.relu.eval()

    def set_as_training(self):
        self.relu.train()

    def change_data_type(self, new_type: DataType):
        self.relu.to(DataType_Torch_Mapping[new_type])

    def set_device(self, device: torch.device):
        self.relu.to(device=device)


class Conv2d_Torch(layer.Conv2d):
    # we can pad the linked block name to the name of layer
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, name: str="Conv2d_Torch", stride: tuple=(1, 1), padding: tuple=(0, 0), dilation: tuple=(1, 1), groups: int=1, bias: bool=True, padding_mode: str='zeros'):
        super(Conv2d_Torch, self).__init__(name)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)

    def forward(self, input_tensor: Tensor_Torch):
        return Tensor_Torch(self.conv2d(input_tensor.get_linked_tensor()), name=self.name + "_output")

    # return KB in memory usage
    def get_memory_size(self):
        return sum([parameter.element_size() * parameter.nelement() for parameter in self.conv2d.parameters()]) / 1024

    def set_as_eval(self):
        self.conv2d.eval()

    def set_as_training(self):
        self.conv2d.eval()

    def change_data_type(self, new_type: DataType):
        self.conv2d.to(DataType_Torch_Mapping[new_type])

    def set_device(self, device: torch.device):
        self.conv2d.to(device=device)





device = torch.device("cuda:0")
m = Conv2d_Torch(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
input = Tensor_Torch(torch.randn(20, 16, 50, 100))

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    input.set_device(device=device)
    m.set_device(device=device)

output = m.forward(input)
print(input.name, "---", input.get_memory_size())
print(m.name, "---", m.get_memory_size())
print(output.name, "---", output.get_memory_size(), output.get_device())


#torch.device("cuda:1")
"""
import inspect
signature = inspect.signature(ReLU_Torch.__init__)
for param in signature.parameters.values():
    print(type(param.name), type(param.default), param.annotation)

dtype = torch.float
device = torch.device("cpu")
D_in, H = 10, 20
w1 = torch.randn(D_in, H, device=device, dtype=dtype)
ReLu = ReLU_Torch()
Tensor = Tensor_Torch(linked_tensor=w1)
output1 = ReLu.forward(Tensor)
print(ReLu.get_memory_size())
"""