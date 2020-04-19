import torch
from PIL import Image
from torchvision.datasets import MNIST
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms

#import os, sys
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from viz_api import input
from viz_api.viz_pytorch_api.tensor import Tensor_Torch

"""
input is the abstraction for the data source abstraction.
As for input, it can be constant tensor, img loader, or the present dataset.
After load, the processed tensor is in the input.
"""
# -------------------------------- Input Node -------------------------------- #

# generate random tensor with shape
class RandomLoader_Torch(input.RandomLoader):
    def __init__(self, view: list, name: str="RandomeLoader_Torch", device=torch.device("cpu")):
        super(RandomLoader_Torch, self).__init__(name)
        self.linked_tensor_group = list()
        self.linked_tensor_group.append(Tensor_Torch(torch.randn(*view, device=device), name=self.name + "_random_tensor_1"))

    def get_loaded_tensor(self):
        return self.linked_tensor_group[0]

    def set_device(self, device: torch.device):
        self.linked_tensor_group[0].set_device(device=device)

    def get_device(self):
        return self.linked_tensor_group[0].get_device()

    # return KB in memory usage for the loaded tensor
    def get_tensor_memory_size(self):
        return self.linked_tensor_group[0].get_self_memory_size()

    # return KB in memory usage for gradients of the loaded tensor
    def get_tensor_grad_memory_size(self):
        return self.linked_tensor_group[0].get_grad_memory_size()

    def remove_from_tracking_gradient(self):
        return self.linked_tensor_group[0].remove_from_tracking_gradient()

    def start_tracking_gradient(self):
        return self.linked_tensor_group[0].start_tracking_gradient()

    @staticmethod
    def get_description(self):
        return "Loader for random tensor"

# generate constant tensor with shape (zero or one)
class ConstantLoader_Torch(input.ConstantLoader):
    def __init__(self, view: list, value: int, name: str="ConstantLoader_Torch", device=torch.device("cpu")):
        super(ConstantLoader_Torch, self).__init__(name)
        self.linked_tensor_group = list()
        self.linked_tensor_group.append(Tensor_Torch(torch.add(torch.zeros(*view, device=device), value), name=self.name + "_const_tensor_1"))

    def get_loaded_tensor(self):
        return self.linked_tensor_group[0]

    def set_device(self, device: torch.device):
        self.linked_tensor_group[0].set_device(device=device)

    def get_device(self):
        return self.linked_tensor_group[0].get_device()

    # return KB in memory usage for the loaded tensor
    def get_tensor_memory_size(self):
        return self.linked_tensor_group[0].get_self_memory_size()

    # return KB in memory usage for gradients of the loaded tensor
    def get_tensor_grad_memory_size(self):
        return self.linked_tensor_group[0].get_grad_memory_size()

    def remove_from_tracking_gradient(self):
        return self.linked_tensor_group[0].remove_from_tracking_gradient()

    def start_tracking_gradient(self):
        return self.linked_tensor_group[0].start_tracking_gradient()

    # dummy return
    def get_number_batch(self):
        return 1

    @staticmethod
    def get_description(self):
        return "Loader for constant tensor (1, 0)"

# load tensor
class TensorLoader_Torch(input.TensorLoader):
    def __init__(self, tensor_path: str, name: str="TensorLoader_Torch", device=torch.device("cpu")):
        super(TensorLoader_Torch, self).__init__(name)
        self.linked_tensor_group = list()
        self.linked_tensor_group.append(Tensor_Torch(torch.load(tensor_path).to(device), name=self.name + "_saved_tensor_1"))

    def get_loaded_tensor(self):
        return self.linked_tensor_group[0]

    def set_device(self, device: torch.device):
        self.linked_tensor_group[0].set_device(device=device)

    def get_device(self):
        return self.linked_tensor_group[0].get_device()

    # return KB in memory usage for the loaded tensor
    def get_tensor_memory_size(self):
        return self.linked_tensor_group[0].get_self_memory_size()

    # return KB in memory usage for gradients of the loaded tensor
    def get_tensor_grad_memory_size(self):
        return self.linked_tensor_group[0].get_grad_memory_size()

    def remove_from_tracking_gradient(self):
        return self.linked_tensor_group[0].remove_from_tracking_gradient()

    def start_tracking_gradient(self):
        return self.linked_tensor_group[0].start_tracking_gradient()

    # dummy return 1 as batch size
    def get_number_batch(self):
        return 1

    @staticmethod
    def get_description(self):
        return "Loader for tensor"

# load one image
class ImageLoader_Torch(input.ImageLoader):
    def __init__(self, image_path: str, imsize: int=512, name: str="ImageLoader_Torch", device=torch.device("cpu")):
        super(ImageLoader_Torch, self).__init__(name)
        loader = transforms.Compose([transforms.Resize(imsize), transforms.ToTensor()])
        image = loader(Image.open(image_path, mode="r")).unsqueeze(0)
        self.linked_tensor_group = list()
        self.linked_tensor_group.append(Tensor_Torch(image.to(device, torch.float), name=self.name + "_image_tensor_1"))

    def get_loaded_tensor(self):
        return self.linked_tensor_group[0]

    def set_device(self, device: torch.device):
        self.linked_tensor_group[0].set_device(device=device)

    def get_device(self):
        return self.linked_tensor_group[0].get_device()

    # return KB in memory usage for the loaded tensor
    def get_tensor_memory_size(self):
        return self.linked_tensor_group[0].get_self_memory_size()

    # return KB in memory usage for gradients of the loaded tensor
    def get_tensor_grad_memory_size(self):
        return self.linked_tensor_group[0].get_grad_memory_size()

    def remove_from_tracking_gradient(self):
        return self.linked_tensor_group[0].remove_from_tracking_gradient()

    def start_tracking_gradient(self):
        return self.linked_tensor_group[0].start_tracking_gradient()

    # dummy return 1 as batch size
    def get_number_batch(self):
        return 1

    @staticmethod
    def get_description(self):
        return "Loader for single image"

# load training set or validation set - this is for mnist dataset, can easy extend (MAKE SURE THE OUTPUT IS torch_tensor)
class MnistDataSetLoader_Torch(input.ImageDataSetLoader):
    def __init__(self, root: str, max_batch_size: int=1, shuffle: bool=False, train: bool=True, download: bool=False, name: str="MnistDataSetLoader_Torch", device=torch.device("cpu")):
        super(MnistDataSetLoader_Torch, self).__init__(name)
        # standard load procedure for MNIST dataset
        mnist_data = MNIST(root, train=train, download=download, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))
        # mnist have 60000 training, 10000 validation, the batch size should divisible by that number
        batch_size = max_batch_size
        image_number = 60000 if train else 10000
        for i in range(max_batch_size, 0, -1):
            if image_number%i == 0:
                batch_size = i
                break
        # load data in batch, which is a iterable list
        mnist_data_loader = DataLoader(mnist_data, batch_size=batch_size, shuffle=shuffle)
        # transfer the data to expected device, pack into Tensor_Torch
        self.linked_tensor_group_img = list()
        self.linked_tensor_group_label = list()
        for i, (images, labels) in enumerate(mnist_data_loader):
            self.linked_tensor_group_img.append(Tensor_Torch(images.to(device), name=self.name + "_img_tensor_" + str(i+1)))
            self.linked_tensor_group_label.append(Tensor_Torch(labels.to(device), name=self.name + "_label_tensor_" + str(i+1)))

    def get_loaded_tensor_img_single(self, index: int=0):
        return self.linked_tensor_group_img[index]

    def get_loaded_tensor_label_single(self, index: int=0):
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

    @staticmethod
    def get_description(self):
        return "Loader for MNIST dataset"


# -------------------------------- Constant Node -------------------------------- #

# generate random tensor with shape
class RandomConstant_Torch(input.RandomConstant):
    def __init__(self, view: list, name: str="RandomeConstant_Torch", device=torch.device("cpu")):
        super(RandomConstant_Torch, self).__init__(name)
        self.linked_tensor_torch = Tensor_Torch(torch.randn(*view, device=device), name=self.name + "_random_tensor")

    def get_saved_tensor(self):
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

    @staticmethod
    def get_description(self):
        return "Random tensor constant"

# generate constant tensor with shape (zero or one)
class ConstantConstant_Torch(input.ConstantConstant):
    def __init__(self, view: list, value: int, name: str = "ConstantConstant_Torch", device=torch.device("cpu")):
        super(ConstantConstant_Torch, self).__init__(name)
        self.linked_tensor_torch = Tensor_Torch(torch.add(torch.zeros(*view, device=device), value),
                                                name=self.name + "const_tensor")

    def get_saved_tensor(self):
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

    @staticmethod
    def get_description(self):
        return "Constant tensor constant (1, 0)"

# load tensor
class TensorConstant_Torch(input.TensorConstant):
    def __init__(self, tensor_path: str, name: str="TensorConstant_Torch", device=torch.device("cpu")):
        super(TensorConstant_Torch, self).__init__(name)
        self.linked_tensor_torch = Tensor_Torch(torch.load(tensor_path).to(device), name=self.name + "_saved_tensor")

    def get_saved_tensor(self):
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

    @staticmethod
    def get_description(self):
        return "Constant Tensor constant"

# load one image
class ImageConstant_Torch(input.ImageConstant):
    def __init__(self, image_path: str, imsize: int = 512, name: str = "ImageConstant_Torch", device=torch.device("cpu")):
        super(ImageConstant_Torch, self).__init__(name)
        loader = transforms.Compose([transforms.Resize(imsize), transforms.ToTensor()])
        image = loader(Image.open(image_path, mode="r")).unsqueeze(0)
        self.linked_tensor_torch = Tensor_Torch(image.to(device, torch.float), name=self.name + "_image_tensor")

    def get_saved_tensor(self):
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

    @staticmethod
    def get_description(self):
        return "Loader for single image"

# --------------------- test input --------------------- #
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

    imageHolder = ImageLoader_Torch(image_path="../../static/img/boat.jpg", imsize=600, device=torch.device("cuda:0"))
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

def test_tensor_loader():
    tensorHolder = TensorLoader_Torch(tensor_path="./test.pt", device=torch.device("cuda:0"))
    print(tensorHolder.get_loaded_tensor().get_view())
    print(tensorHolder.get_loaded_tensor().get_linked_tensor())
    print(tensorHolder.get_tensor_memory_size(), tensorHolder.get_tensor_grad_memory_size())
    print(tensorHolder.get_device(), tensorHolder.get_loaded_tensor().get_device())

def test_img_dataset_loader():
    root = "../../static/dataset"
    train = True
    download = True
    imageDataSetHolder = MnistDataSetLoader_Torch(root=root, batch_size=64, shuffle=True, train=train, download=download, device=torch.device("cuda:0"))
    print(imageDataSetHolder.get_number_batch())
    print(imageDataSetHolder.get_tensor_memory_size(), imageDataSetHolder.get_tensor_grad_memory_size())
    print(imageDataSetHolder.get_loaded_tensor_img_single(5).get_self_memory_size(),
          imageDataSetHolder.get_loaded_tensor_label_single(5).get_self_memory_size())
    print(imageDataSetHolder.get_device())
    print(imageDataSetHolder.get_loaded_tensor_img_single(5).get_device(),
          imageDataSetHolder.get_loaded_tensor_label_single(5).get_device())

# --------------------- test constant --------------------- #
def test_img_constant():
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

    imageHolder = ImageConstant_Torch(image_path="../../static/img/boat.jpg", imsize=600, device=torch.device("cuda:0"))
    imshow(imageHolder.get_saved_tensor().get_linked_tensor(), title='Style Image')
    plt.figure()
    print(imageHolder.get_tensor_memory_size(), imageHolder.get_tensor_grad_memory_size())
    print(imageHolder.get_device(), imageHolder.get_saved_tensor().get_device())

def test_random_constant():
    randomHolder = RandomConstant_Torch([10, 20, 30, 40], device=torch.device("cuda:0"))
    print(randomHolder.get_saved_tensor().get_view())
    print(randomHolder.get_tensor_memory_size(), randomHolder.get_tensor_grad_memory_size())
    print(randomHolder.get_device(), randomHolder.get_saved_tensor().get_device())

def test_const_constant():
    constHolder = ConstantConstant_Torch([5, 5, 5, 5], value=5, device=torch.device("cuda:0"))
    print(constHolder.get_saved_tensor().get_view())
    print(constHolder.get_saved_tensor().get_linked_tensor())
    print(constHolder.get_tensor_memory_size(), constHolder.get_tensor_grad_memory_size())
    print(constHolder.get_device(), constHolder.get_saved_tensor().get_device())

def test_tensor_constant():
    tensorHolder = TensorConstant_Torch(tensor_path="./test.pt", device=torch.device("cuda:0"))
    print(tensorHolder.get_saved_tensor().get_view())
    print(tensorHolder.get_saved_tensor().get_linked_tensor())
    print(tensorHolder.get_tensor_memory_size(), tensorHolder.get_tensor_grad_memory_size())
    print(tensorHolder.get_device(), tensorHolder.get_saved_tensor().get_device())


#test_random_loader()
#test_random_constant()
#test_tensor_loader()
#test_tensor_constant()
#test_const_loader()
#test_const_constant()
#test_img_loader()
#test_img_constant()
#test_img_dataset_loader()