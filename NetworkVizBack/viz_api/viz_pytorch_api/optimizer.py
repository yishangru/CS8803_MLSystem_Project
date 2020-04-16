from viz_api.viz_pytorch_api.tensor import Tensor_Torch
from viz_api.layer import LayerWrapper
from viz_api import optimizer
import torch.optim as optim

"""
Optimizer will take the layer and tensor as input. For layer, optimizer will extract the parameters for optimizing.
"""
class SGD_Torch(optimizer.SGD):
    def __init__(self, learning_rate: float, name:str = "SGD_Torch"):
        super(SGD_Torch, self).__init__(name)
        self.optimizer = optim.SGD(lr=learning_rate)
        self.linked_loss = None

    def get_optimizer(self):
        return self.optimizer

    def link_loss_tensor(self, tensor: Tensor_Torch):
        self.linked_loss = tensor

    def register_optimizer(self, object_to_track, learning_rate: float=None):
        # check type for object_to_track - object_to_track (node)
        """
        optim.add_param_group({ 'params': parameters, 'lr': ...})
        """
        self.optimizer.add_param_group()

    def step(self):
        self.optimizer.step()

    def backward(self):
        self.linked_loss.get_linked_tensor().backward()

    def clear_gradient(self):
        self.optimizer.zero_grad()

    def get_despcription(self):
        return "SGD Optimizer"


class LBFGS_Torch(optimizer.LBFGS):
    def __init__(self, learning_rate: float=1, name:str = "LBFGS_Torch"):
        super(LBFGS_Torch, self).__init__(name)
        self.optimizer = optim.SGD(lr=learning_rate)
        self.linked_loss = None

    def get_optimizer(self):
        return self.optimizer

    def link_loss_tensor(self, tensor: Tensor_Torch):
        self.linked_loss = tensor

    def register_optimizer(self, object_to_track, learning_rate: float=None):
        # check type for object_to_track - object_to_track (node)
        """
            optim.add_param_group({ 'params': parameters, 'lr': ...})
        """
        if isinstance(object_to_track, LayerWrapper):
            if learning_rate is None:
                self.optimizer.add_param_group({'params': object_to_track.get_layer().parameters()})
            else:
                self.optimizer.add_param_group({'params': object_to_track.get_layer().parameters(), 'lr': learning_rate})
        elif isinstance(object_to_track,Tensor_Torch):
            if learning_rate is None:
                self.optimizer.add_param_group({'params': object_to_track.get_linked_tensor()})
            else:
                self.optimizer.add_param_group({'params': object_to_track.get_linked_tensor(), 'lr': learning_rate})

    def step(self):
        self.optimizer.step()

    def backward(self):
        self.linked_loss.get_linked_tensor().backward()

    def clear_gradient(self):
        self.optimizer.zero_grad()

    def get_despcription(self):
        return "LBFGS Optimizer"


