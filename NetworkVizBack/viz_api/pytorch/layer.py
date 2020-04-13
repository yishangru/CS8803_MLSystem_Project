import torch
import torch.nn as nn

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viz_api import layer
from viz_api.pytorch.tensor import Tensor_Torch

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
        return self.relu(input_tensor.get_linked_tensor())

    def get_memory_size(self): # relu is 0 in pytorch
        return sum([parameter.element_size() * parameter.nelement() for parameter in self.relu.parameters()])





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