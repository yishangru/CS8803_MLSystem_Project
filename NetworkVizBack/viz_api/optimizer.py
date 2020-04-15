from enum import Enum, auto

class OptimizerType(Enum):
    SGD = auto()    # load single image
     = auto()   # generate random tensor with shape
    ConstantLoader = auto() # generate constant tensor with shape (zero or one)