from viz_api import transform
from viz_api.viz_pytorch_api.tensor import Tensor_Torch

class FlatTransform_Torch(transform.FlatTransform):
    def __init__(self, inplace_forward: bool=False, name: str="FlatTransform_Torch"):
        super(FlatTransform_Torch, self).__init__(name=name)
        self.inplace_forward = inplace_forward

    def forward(self, input_tensor: Tensor_Torch):
        input_linked_tensor = input_tensor.get_linked_tensor()
        output_linked_tensor = input_linked_tensor.view(input_linked_tensor.shape[0], -1)
        if self.inplace_forward:
            input_tensor.set_linked_tensor(output_linked_tensor)
        return Tensor_Torch(output_linked_tensor, name=self.name + "_output")

    def get_description(self):
        return "Flat Tensor to One"


class DataClampTransform_Torch(transform.DataClampTransform):
    def __init__(self, clamp_range: tuple, inplace_forward: bool=False, name: str="DataClampTransform_Torch"):
        super(DataClampTransform_Torch, self).__init__(name=name)
        self.inplace_forward = inplace_forward
        self.clamp_range = clamp_range

    def forward(self, input_tensor: Tensor_Torch):
        input_linked_tensor = input_tensor.get_linked_tensor()
        if self.inplace_forward:
            output_linked_tensor = input_linked_tensor.view(input_linked_tensor.shape[0], -1)
            input_tensor.set_linked_tensor(output_linked_tensor)
        else:
            output_linked_tensor = in
        return Tensor_Torch(output_linked_tensor, name=self.name + "_output")

    def get_description(self):
        return "Flat Tensor to One"