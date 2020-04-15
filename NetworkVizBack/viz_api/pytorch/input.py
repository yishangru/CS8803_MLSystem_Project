import torch
from PIL import Image
import torchvision.transforms as transforms

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viz_api import input
from viz_api.pytorch.tensor import Tensor_Torch
from viz_api.tensor import DataType
from viz_api.pytorch.tensor import DataType_Torch_Mapping

"""
input is the abstraction for the data source abstraction.
As for input, it can be constant tensor, img loader, or the present dataset.
After load, the processed tensor is in the input.
"""

# load one image
class ImageLoader_Torch(input.ImageLoader):
    def __init__(self, image_path: str, imsize: int=512, name: str="ImageLoader_Torch"):
        super(ImageLoader_Torch, self).__init__(name)
        loader = transforms.Compose([transforms.Resize(imsize), transforms.ToTensor()])
        image = Image.open(image_path, mode="r")
        self.linked_tensor_torch = Tensor_Torch(loader(image).unsqueeze(0), name=name + "_image_tensor")
        self.device = torch.device("cpu")

    def get_loaded_tensor(self):
        return self.linked_tensor_torch

    def set_device(self, device: torch.device):
        self.linked_tensor_torch.set_device(device=device)
        self.device = device

    def get_device(self):
        return self.device

    # return KB in memory usage for the loaded tensor
    def get_tensor_memory_size(self):
        return self.linked_tensor_torch.get_self_memory_size()

    # return KB in memory usage for gradients of the loaded tensor
    def get_tensor_grad_memory_size(self):
        return self.linked_tensor_torch.get_grad_memory_size()

    def get_despcription(self):
        return "Loader for single image"

# generate random tensor with shape
class RandomLoader_Torch(input.RandomLoader):
    def __init__(self, name: str="RandomeLoader_Torch"):
        super(RandomLoader_Torch, self).__init__(name)

    def load_tensor(self):
        pass

    def get_loaded_tensor(self):
        pass

    def get_despcription(self):
        return "Loader for random tensor"

# load training set or validation set
class ImageDateSetLoader_Torch(input.ImageDateSetLoader):
    def __init__(self, name: str="ImageDataSetLoader_Torch"):
        super(ImageDateSetLoader_Torch, self).__init__(name)

    # return KB in memory usage for the loaded tensor
    def get_tensor_memory_size(self):
        return

    # return KB in memory usage for gradients of the loaded tensor
    def get_tensor_grad_memory_size(self):
        return NotImplementedError

    def get_loaded_tensor(self):
        pass

    def get_despcription(self):
        return "Loader for image dataset (torchvision)"

# generate constant tensor with shape (zero or one)
class ConstantLoader_Torch(input.ConstantLoader):
    def __init__(self, name: str="ConstantLoader_Torch"):
        super(ConstantLoader_Torch, self).__init__(name)

    def get_loaded_tensor(self):
        pass

    def get_despcription(self):
        return "Loader for constant tensor (1, 0)"



import matplotlib.pyplot as plt
unloader = transforms.ToPILImage()  # reconvert into PIL image
plt.ion()
def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

imageHolder = ImageLoader_Torch(image_path="../../img/boat.jpg", imsize=600)
imshow(imageHolder.get_loaded_tensor().get_linked_tensor(), title='Style Image')
plt.figure()
print(imageHolder.get_tensor_memory_size(), imageHolder.get_tensor_grad_memory_size())