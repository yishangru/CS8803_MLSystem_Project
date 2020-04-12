import torch
from platform.layerWrapper import layerWrapper

"""
layer is the abstraction for the underlying platform API.
As for layer, it is composed of a nn.module, input source,
output source and the parent block.

If no parent block, it will link with the upper model. We can add transform for
the input defined in the layer level. There is only one input and output source.
"""

class layer(layerWrapper):
    def __init__(self):
        super(layer, self).__init__()
        