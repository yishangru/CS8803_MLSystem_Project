from enum import Enum, auto

# -------------------------------- Input Node -------------------------------- #
class InputType(Enum):
    ImageLoader = auto()    # load single image as input
    RandomLoader = auto()   # generate random tensor with shape as input
    ConstantLoader = auto() # generate constant tensor with shape (zero or one) as input
    TensorLoader = auto()   # load tensor from location as input
    ImageDataSetLoader = auto()  # load image dataset

class ImageDataSetType(Enum):
    MnistLoader = auto()    # load Mnist dataset

# data loader, it will contain a batch tensor which holder the data
class InputWrapper(object):
    def __init__(self, name: str):
        super(InputWrapper, self).__init__()
        self.name = name

    def set_device(self, device):
        raise NotImplementedError

    def get_device(self):
        raise NotImplementedError

    # return KB in memory usage for the loaded tensor
    def get_tensor_memory_size(self):
        raise NotImplementedError

    # return KB in memory usage for gradients of the loaded tensor
    def get_tensor_grad_memory_size(self):
        raise NotImplementedError

    def remove_from_tracking_gradient(self):
        raise NotImplementedError

    def start_tracking_gradient(self):
        raise NotImplementedError

    # return how many batch in the input, useful for dataset
    def get_number_batch(self):
        raise NotImplementedError

    def get_despcription(self):
        raise NotImplementedError

# load one image
class ImageLoader(InputWrapper):
    def __init__(self, name: str):
        super(ImageLoader, self).__init__(name)

# generate random tensor with shape
class RandomLoader(InputWrapper):
    def __init__(self, name: str):
        super(RandomLoader, self).__init__(name)

# generate constant tensor with shape (zero or one)
class ConstantLoader(InputWrapper):
    def __init__(self, name: str):
        super(ConstantLoader, self).__init__(name)

# load tensor as input
class TensorLoader(InputWrapper):
    def __init__(self, name: str):
        super(TensorLoader, self).__init__(name)

# load training set or validation set for image data set
class ImageDataSetLoader(InputWrapper):
    def __init__(self, name: str):
        super(ImageDataSetLoader, self).__init__(name)


# -------------------------------- Constant Node -------------------------------- #
class ConstantType(Enum):
    ImageConstant = auto()
    RandomConstant = auto()
    ConstantConstant = auto()
    TensorConstant = auto()

# data loader, it will contain a tensor which holder the data
class ConstantWrapper(object):
    def __init__(self, name: str):
        super(ConstantWrapper, self).__init__()
        self.name = name

    def get_saved_tensor(self):
        raise NotImplementedError

    def set_device(self, device):
        raise NotImplementedError

    def get_device(self):
        raise NotImplementedError

    # return KB in memory usage for the loaded tensor
    def get_tensor_memory_size(self):
        raise NotImplementedError

    # return KB in memory usage for gradients of the loaded tensor
    def get_tensor_grad_memory_size(self):
        raise NotImplementedError

    def remove_from_tracking_gradient(self):
        raise NotImplementedError

    def start_tracking_gradient(self):
        raise NotImplementedError

    def get_despcription(self):
        raise NotImplementedError

# load one image
class ImageConstant(ConstantWrapper):
    def __init__(self, name: str):
        super(ImageConstant, self).__init__(name)

# generate random tensor with shape
class RandomConstant(ConstantWrapper):
    def __init__(self, name: str):
        super(RandomConstant, self).__init__(name)

# generate constant tensor with shape (zero or one)
class ConstantConstant(ConstantWrapper):
    def __init__(self, name: str):
        super(ConstantConstant, self).__init__(name)

# load tensor as input
class TensorConstant(ConstantWrapper):
    def __init__(self, name: str):
        super(TensorConstant, self).__init__(name)