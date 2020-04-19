import torch
import torchvision
from viz_api import transform
from viz_api.viz_pytorch_api.tensor import Tensor_Torch

"""
To do:
1. Change the single tensor operation to calling underlying Tensor_Torch method (need add methods in Tensor_Torch)
"""

class FlatTransform_Torch(transform.FlatTransform):
    def __init__(self, inplace_forward: bool=False, name: str="FlatTransform_Torch"):
        super(FlatTransform_Torch, self).__init__(name=name)
        self.inplace_forward = inplace_forward

    def forward(self, input_tensor: Tensor_Torch):
        input_linked_tensor = input_tensor.get_linked_tensor()
        output_linked_tensor = input_linked_tensor.view(input_linked_tensor.shape[0], -1)
        if self.inplace_forward:
            input_tensor.set_linked_tensor(output_linked_tensor)
        return output_linked_tensor

    def get_description(self):
        return "Flat tensor to One"


class NormalizeTransform_Torch(transform.NormalizeTransform):
    def __init__(self, mean: tuple=(0.485, 0.456, 0.406), std: tuple=(0.229, 0.224, 0.225), inplace_forward: bool=False, name: str="NormalizeTransform_Torch"):
        super(NormalizeTransform_Torch, self).__init__(name=name)
        self.inplace_forward = inplace_forward
        self.mean = torch.tensor(list(mean)).view(-1, 1, 1)
        self.std = torch.tensor(list(std)).view(-1, 1, 1)

    def forward(self, input_tensor: Tensor_Torch):
        self.mean = self.mean.to(input_tensor.get_device())
        self.std = self.std.to(input_tensor.get_device())
        input_linked_tensor = input_tensor.get_linked_tensor()
        if self.inplace_forward:
            input_linked_tensor.sub_(self.mean).div_(self.std)
            return input_linked_tensor
        else:
            output_linked_tensor = input_linked_tensor.sub(self.mean).div_(self.std)
            return output_linked_tensor

    def get_description(self):
        return "Normalize img tensor to certain range"


class DataClampTransform_Torch(transform.DataClampTransform):
    def __init__(self, clamp_range: tuple=(0, 1), inplace_forward: bool=False, name: str="DataClampTransform_Torch"):
        super(DataClampTransform_Torch, self).__init__(name=name)
        self.inplace_forward = inplace_forward
        self.clamp_range = clamp_range

    def forward(self, input_tensor: Tensor_Torch):
        input_linked_tensor = input_tensor.get_linked_tensor()
        if self.inplace_forward:
            input_linked_tensor.clamp_(self.clamp_range[0], self.clamp_range[1])
            return input_linked_tensor
        else:
            output_linked_tensor = input_linked_tensor.clamp(self.clamp_range[0], self.clamp_range[1])
            return output_linked_tensor

    def get_description(self):
        return "Clamp tensor to certain range"


class DetachTransform_Torch(transform.DetachTransform):
    def __init__(self, inplace_forward: bool=False, name: str="DetachTransform_Torch"):
        super(DetachTransform_Torch, self).__init__(name=name)
        self.inplace_forward = inplace_forward

    """
    detach() will return a tensor share the memory, so any following inplace operation will effect on both tensor
    _detach() is the inplace version for detach(); Thus, detach will always share the memory.
    normal operation: clone and then detach
    """
    def forward(self, input_tensor: Tensor_Torch):
        input_linked_tensor = input_tensor.get_linked_tensor()
        if self.inplace_forward:
            input_linked_tensor.detach_()
            return input_linked_tensor
        else:
            output_linked_tensor = input_linked_tensor.detach()
            return output_linked_tensor

    def get_description(self):
        return "Detach tensor from computation graph. When use detach operation, " \
               "it is safer to first clone the tensor and then use the inplace detach version"


class AddTransform_Torch(transform.AddTransform):
    def __init__(self, inplace_forward: bool=False, name: str="AddTransform_Torch"):
        super(AddTransform_Torch, self).__init__(name=name)
        self.inplace_forward = inplace_forward

    """
    inplace version will update the first tensor inplace
    """
    def forward(self, *input_tensor):
        input_linked_tensor_list = list(input_tensor)
        output_linked_tensor = input_linked_tensor_list[0].get_linked_tensor() if self.inplace_forward else input_linked_tensor_list[0].get_deep_copy()
        for i in range(1, len(input_linked_tensor_list)):
            output_linked_tensor.add_(input_linked_tensor_list[i].get_linked_tensor())
        return output_linked_tensor

    def get_description(self):
        return "Add multiple tensors"


class GetGramMatrix_Torch(transform.GetGramMatrix):
    def __init__(self, inplace_forward: bool=False, name: str="AddTransform_Torch"):
        super(GetGramMatrix_Torch, self).__init__(name=name)
        self.inplace_forward = inplace_forward

    def forward(self, input_tensor: Tensor_Torch):
        input_linked_tensor = input_tensor.get_linked_tensor()
        # reference in Pytorch Style Transfer Tutorial
        a, b, c, d = input_linked_tensor.size() # should be as four tuple
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)
        features = input_linked_tensor.view(a * b, c * d)  # resise F_XL into \hat F_XL
        G = torch.mm(features, features.t())  # compute the gram product
        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        output_linked_tensor = G.div(a * b * c * d)
        if self.inplace_forward:
            input_tensor.set_linked_tensor(output_linked_tensor)
        return output_linked_tensor

    def get_description(self):
        return "Get Gram Features (for image)"

# --------------------- test input --------------------- #
def test_flat_transform():
    from viz_api.viz_pytorch_api.input import ImageConstant_Torch
    image_tensor_input = ImageConstant_Torch(image_path="../../static/img/boat.jpg", imsize=512, device=torch.device("cuda:0")).get_saved_tensor()
    flat_transform = FlatTransform_Torch(inplace_forward=True)
    image_tensor_output = Tensor_Torch(flat_transform.forward(image_tensor_input))
    print(image_tensor_input.get_linked_tensor().size(), image_tensor_output.get_linked_tensor().size())
    print(image_tensor_input.get_device(), image_tensor_output.get_device())

def test_normal_transform():
    from viz_api.viz_pytorch_api.input import ImageConstant_Torch
    image_tensor_input = ImageConstant_Torch(image_path="../../static/img/boat.jpg", imsize=512, device=torch.device("cuda:0")).get_saved_tensor()
    normal_transform = NormalizeTransform_Torch(inplace_forward=True)
    image_tensor_output = Tensor_Torch(normal_transform.forward(image_tensor_input))
    print(image_tensor_input.get_linked_tensor().size(), image_tensor_output.get_linked_tensor().size())
    print(image_tensor_input.get_device(), image_tensor_output.get_device())

def test_data_clamp_transform():
    rand_tensor_input = Tensor_Torch(torch.randn(3, 2).add(5))
    clamp_transform = DataClampTransform_Torch((0, 1), inplace_forward=True)
    rand_tensor_output = Tensor_Torch(clamp_transform.forward(rand_tensor_input))
    print(rand_tensor_output.get_linked_tensor())
    print(torch.eq(rand_tensor_input.get_linked_tensor(), rand_tensor_output.get_linked_tensor()))

    rand_tensor_input = Tensor_Torch(torch.randn(3, 2).add(5))
    clamp_transform = DataClampTransform_Torch((0, 1), inplace_forward=False)
    rand_tensor_output = Tensor_Torch(clamp_transform.forward(rand_tensor_input))
    print(rand_tensor_output.get_linked_tensor())
    print(torch.eq(rand_tensor_input.get_linked_tensor(), rand_tensor_output.get_linked_tensor()))

def test_detach_transform():
    rand_tensor_input = Tensor_Torch(torch.randn(3, 2).requires_grad_(True))
    detach_transform = DetachTransform_Torch(inplace_forward=True)
    rand_tensor_output = Tensor_Torch(detach_transform.forward(rand_tensor_input))
    print(rand_tensor_input.get_linked_tensor().requires_grad, rand_tensor_output.get_linked_tensor().requires_grad)

    rand_tensor_input = Tensor_Torch(torch.randn(3, 2).requires_grad_(True))
    detach_transform = DetachTransform_Torch(inplace_forward=False)
    rand_tensor_output = Tensor_Torch(detach_transform.forward(rand_tensor_input))
    print(rand_tensor_input.get_linked_tensor().requires_grad, rand_tensor_output.get_linked_tensor().requires_grad)

def test_add_transform():
    one_tensor_input_1 = Tensor_Torch(torch.ones(1, 1))
    one_tensor_input_2 = Tensor_Torch(torch.ones(1, 1))
    one_tensor_input_3 = Tensor_Torch(torch.ones(1, 1))
    add_transform = AddTransform_Torch(inplace_forward=True)
    one_tensor_output = Tensor_Torch(add_transform.forward(one_tensor_input_1, one_tensor_input_2, one_tensor_input_3))
    print(one_tensor_input_1.get_linked_tensor(), one_tensor_input_2.get_linked_tensor(),
          one_tensor_input_3.get_linked_tensor(), one_tensor_output.get_linked_tensor())

    one_tensor_input_1 = Tensor_Torch(torch.ones(1, 1))
    one_tensor_input_2 = Tensor_Torch(torch.ones(1, 1))
    one_tensor_input_3 = Tensor_Torch(torch.ones(1, 1))
    add_transform = AddTransform_Torch(inplace_forward=False)
    one_tensor_output = Tensor_Torch(add_transform.forward(one_tensor_input_1, one_tensor_input_2, one_tensor_input_3))
    print(one_tensor_input_1.get_linked_tensor(), one_tensor_input_2.get_linked_tensor(),
          one_tensor_input_3.get_linked_tensor(), one_tensor_output.get_linked_tensor())

def test_gram_matrix_transform():
    from viz_api.viz_pytorch_api.input import ImageConstant_Torch
    image_tensor_input = ImageConstant_Torch(image_path="../../static/img/boat.jpg", imsize=512,
                                             device=torch.device("cuda:0")).get_saved_tensor()
    gram_transform = GetGramMatrix_Torch(inplace_forward=True)
    gram_tensor_output = Tensor_Torch(gram_transform.forward(image_tensor_input))
    print(image_tensor_input.get_linked_tensor().size(), gram_tensor_output.get_linked_tensor().size())
    print(image_tensor_input.get_device(), gram_tensor_output.get_device())

    from viz_api.viz_pytorch_api.input import ImageConstant_Torch
    image_tensor_input = ImageConstant_Torch(image_path="../../static/img/boat.jpg", imsize=512,
                                             device=torch.device("cuda:0")).get_saved_tensor()
    gram_transform = GetGramMatrix_Torch(inplace_forward=False)
    gram_tensor_output = Tensor_Torch(gram_transform.forward(image_tensor_input))
    print(image_tensor_input.get_linked_tensor().size(), gram_tensor_output.get_linked_tensor().size())
    print(image_tensor_input.get_device(), gram_tensor_output.get_device())

#test_flat_transform()
test_normal_transform()
#test_data_clamp_transform()
#test_detach_transform()
#test_add_transform()
#test_gram_matrix_transform()