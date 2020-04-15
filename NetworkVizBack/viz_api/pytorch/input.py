import torch
from PIL import Image
from torchvision.datasets import MNIST
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viz_api import input
from viz_api.pytorch.tensor import Tensor_Torch

"""
input is the abstraction for the data source abstraction.
As for input, it can be constant tensor, img loader, or the present dataset.
After load, the processed tensor is in the input.
"""

# generate random tensor with shape
class RandomLoader_Torch(input.RandomLoader):
    def __init__(self, view: list, name: str="RandomeLoader_Torch", device=torch.device("cpu")):
        super(RandomLoader_Torch, self).__init__(name)
        self.linked_tensor_torch = Tensor_Torch(torch.randn(*view, device=device), name=self.name + "_random_tensor_1")

    def get_loaded_tensor(self):
        return self.linked_tensor_torch

    def set_device(self, device: torch.device):
        self.linked_tensor_torch.set_device(device=device)

    def get_device(self):
        return self.linked_tensor_torch.get_device()

    # return KB in memory usage for the loaded tensor
    def get_tensor_memory_size(self):
        return self.linked_tensor_torch.get_self_memory_size()

    # return KB in memory usage for gradients of the loaded tensor
    def get_tensor_grad_memory_size(self):
        return self.linked_tensor_torch.get_grad_memory_size()

    def remove_from_tracking_gradient(self):
        return self.linked_tensor_torch.remove_from_tracking_gradient()

    def start_tracking_gradient(self):
        return self.linked_tensor_torch.start_tracking_gradient()

    def get_despcription(self):
        return "Loader for random tensor"

# generate constant tensor with shape (zero or one)
class ConstantLoader_Torch(input.ConstantLoader):
    def __init__(self, view: list, value: int, name: str="ConstantLoader_Torch", device=torch.device("cpu")):
        super(ConstantLoader_Torch, self).__init__(name)
        self.linked_tensor_torch = Tensor_Torch(torch.add(torch.zeros(*view, device=device), value), name=self.name + "const_tensor")

    def get_loaded_tensor(self):
        return self.linked_tensor_torch

    def set_device(self, device: torch.device):
        self.linked_tensor_torch.set_device(device=device)

    def get_device(self):
        return self.linked_tensor_torch.get_device()

    # return KB in memory usage for the loaded tensor
    def get_tensor_memory_size(self):
        return self.linked_tensor_torch.get_self_memory_size()

    # return KB in memory usage for gradients of the loaded tensor
    def get_tensor_grad_memory_size(self):
        return self.linked_tensor_torch.get_grad_memory_size()

    def remove_from_tracking_gradient(self):
        return self.linked_tensor_torch.remove_from_tracking_gradient()

    def start_tracking_gradient(self):
        return self.linked_tensor_torch.start_tracking_gradient()

    def get_despcription(self):
        return "Loader for constant tensor (1, 0)"

# load one image
class ImageLoader_Torch(input.ImageLoader):
    def __init__(self, image_path: str, imsize: int=512, name: str="ImageLoader_Torch", device=torch.device("cpu")):
        super(ImageLoader_Torch, self).__init__(name)
        loader = transforms.Compose([transforms.Resize(imsize), transforms.ToTensor()])
        image = loader(Image.open(image_path, mode="r")).unsqueeze(0)
        self.linked_tensor_torch = Tensor_Torch(image.to(device, torch.float), name=self.name + "_image_tensor_1")

    def get_loaded_tensor(self):
        return self.linked_tensor_torch

    def set_device(self, device: torch.device):
        self.linked_tensor_torch.set_device(device=device)

    def get_device(self):
        return self.linked_tensor_torch.get_device()

    # return KB in memory usage for the loaded tensor
    def get_tensor_memory_size(self):
        return self.linked_tensor_torch.get_self_memory_size()

    # return KB in memory usage for gradients of the loaded tensor
    def get_tensor_grad_memory_size(self):
        return self.linked_tensor_torch.get_grad_memory_size()

    def remove_from_tracking_gradient(self):
        return self.linked_tensor_torch.remove_from_tracking_gradient()

    def start_tracking_gradient(self):
        return self.linked_tensor_torch.start_tracking_gradient()

    def get_despcription(self):
        return "Loader for single image"

# load training set or validation set - this is for mnist dataset, can easy extend (MAKE SURE THE OUTPUT IS torch_tensor)
class MnistDataSetLoader_Torch(input.ImageDataSetLoader):
    def __init__(self, root: str, batch_size: int=1, shuffle: bool=False, train: bool=True, download: bool=False, name: str="MnistDataSetLoader_Torch", device=torch.device("cpu")):
        super(MnistDataSetLoader_Torch, self).__init__(name)
        # standard load procedure for MNIST dataset
        mnist_data = MNIST(root, train=train, download=download, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))
        mnist_data_loader = DataLoader(mnist_data, batch_size=batch_size, shuffle=shuffle)  # this is a iterable list
        # transfer the data to expected device, pack into Tensor_Torch
        self.linked_tensor_group_img = list()
        self.linked_tensor_group_label = list()
        for i, (images, labels) in enumerate(mnist_data_loader):
            self.linked_tensor_group_img.append(Tensor_Torch(images.to(device), name=self.name + "_img_tensor_" + str(i+1)))
            self.linked_tensor_group_label.append(Tensor_Torch(labels.to(device), name=self.name + "_label_tensor_" + str(i+1)))

    # dummy return
    def get_loaded_tensor(self):
        return self.linked_tensor_group_img[0]

    def get_loaded_tensor_img(self, index: int=0):
        return self.linked_tensor_group_img[index]

    def get_loaded_tensor_label(self, index: int=0):
        return self.linked_tensor_group_label[index]

    def set_device(self, device: torch.device):
        for counter in range(len(self.linked_tensor_group_img)):
            self.linked_tensor_group_img[counter].set_device(device=device)
            self.linked_tensor_group_label[counter].set_device(device=device)

    # use the first tensor device as the device for group
    def get_device(self):
        return self.linked_tensor_group_img[0].get_device()

    # return KB in memory usage for the loaded tensor
    def get_tensor_memory_size(self):
        return sum([img.get_self_memory_size() for img in self.linked_tensor_group_img]) \
               + sum([label.get_self_memory_size() for label in self.linked_tensor_group_label])

    # return KB in memory usage for gradients of the loaded tensor
    def get_tensor_grad_memory_size(self):  # we would only track the grad of img
        return sum([img.get_grad_memory_size() for img in self.linked_tensor_group_img])

    def remove_from_tracking_gradient(self):
        for counter in range(len(self.linked_tensor_group_img)):
            self.linked_tensor_group_img[counter].remove_from_tracking_gradient()

    def start_tracking_gradient(self):
        for counter in range(len(self.linked_tensor_group_img)):
            self.linked_tensor_group_img[counter].start_tracking_gradient()

    # return how many batch in the input, useful for dataset
    def get_number_batch(self):
        return len(self.linked_tensor_group_img)

    def get_despcription(self):
        return "Loader for MNIST dataset"

# --------------------- test --------------------- #
def test_img_loader():
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

    imageHolder = ImageLoader_Torch(image_path="../../img/boat.jpg", imsize=600, device=torch.device("cuda:0"))
    imshow(imageHolder.get_loaded_tensor().get_linked_tensor(), title='Style Image')
    plt.figure()
    print(imageHolder.get_tensor_memory_size(), imageHolder.get_tensor_grad_memory_size())
    print(imageHolder.get_device(), imageHolder.get_loaded_tensor().get_device())

def test_random_loader():
    randomHolder = RandomLoader_Torch([10, 20, 30, 40], device=torch.device("cuda:0"))
    print(randomHolder.get_loaded_tensor().get_view())
    print(randomHolder.get_tensor_memory_size(), randomHolder.get_tensor_grad_memory_size())
    print(randomHolder.get_device(), randomHolder.get_loaded_tensor().get_device())

def test_const_loader():
    constHolder = ConstantLoader_Torch([5, 5, 5, 5], value=5, device=torch.device("cuda:0"))
    print(constHolder.get_loaded_tensor().get_view())
    print(constHolder.get_loaded_tensor().get_linked_tensor())
    print(constHolder.get_tensor_memory_size(), constHolder.get_tensor_grad_memory_size())
    print(constHolder.get_device(), constHolder.get_loaded_tensor().get_device())

def test_img_dataset_loader():
    root = "../../dataset"
    train = True
    download = True
    imageDataSetHolder = MnistDataSetLoader_Torch(root=root, batch_size=64, shuffle=True, train=train, download=download, device=torch.device("cuda:0"))
    print(imageDataSetHolder.get_number_batch())
    print(imageDataSetHolder.get_tensor_memory_size(), imageDataSetHolder.get_tensor_grad_memory_size())
    print(imageDataSetHolder.get_loaded_tensor().get_self_memory_size(),
          imageDataSetHolder.get_loaded_tensor_img(5).get_self_memory_size(),
          imageDataSetHolder.get_loaded_tensor_label(5).get_self_memory_size())
    print(imageDataSetHolder.get_device())
    print(imageDataSetHolder.get_loaded_tensor().get_device(),
          imageDataSetHolder.get_loaded_tensor_img(5).get_device(),
          imageDataSetHolder.get_loaded_tensor_label(5).get_device())