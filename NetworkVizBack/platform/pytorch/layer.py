import torch
import inspect
import torch.nn as nn

import abstraction.layer as layer
from

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

    def forward(self, input: ):
        return self.relu(input)

signature = inspect.signature(ReLU_Torch.__init__)
for name, parameter in signature.items():
    print(name, parameter.default, parameter.annotation, parameter.kind)