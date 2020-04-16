from enum import Enum, auto

class OptimizerType(Enum):
    SGD = auto()
    LBFGS = auto()

"""
Optimizer contain a loss function for taking input and use that input for update gradient.
"""

class OptimizerWrapper(object):
    def __init__(self, name: str):
        super(OptimizerWrapper, self).__init__()
        self.name = name

    def get_optimizer(self):
        raise NotImplementedError

    def link_loss_tensor(self, tensor):
        raise NotImplementedError

    def register_optimizer(self, object_to_track):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def clear_gradient(self):
        raise NotImplementedError

    def get_despcription(self):
        raise NotImplementedError


class SGD(OptimizerWrapper):
    def __init__(self, name: str):
        super(SGD, self).__init__(name)


class LBFGS(OptimizerWrapper):
    def __init__(self, name: str):
        super(LBFGS, self).__init__(name)