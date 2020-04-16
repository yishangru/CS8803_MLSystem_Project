import torch
from viz_api import transform
from viz_api.viz_pytorch_api.tensor import Tensor_Torch

class FlatTransform_Torch(transform.FlatTransform):
    def __init__(self, inplace_forward: bool=False, name: str="FlatTransform_Torch"):
        super(FlatTransform_Torch, self).__init__(name=name)
        self.inplace_forward = inplace_forward

    def forward(self, ):