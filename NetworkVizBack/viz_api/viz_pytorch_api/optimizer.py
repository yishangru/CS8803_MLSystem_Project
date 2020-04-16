import torch.optim as optim
from viz_api import optimizer
from viz_api.viz_pytorch_api.tensor import Tensor_Torch
from viz_api.layer import LayerWrapper

"""
Optimizer will take the layer and tensor as input. For layer, optimizer will extract the parameters for optimizing.
"""
class SGD_Torch(optimizer.SGD):
    def __init__(self, object_to_track_list: list, learning_rate: float, momentum: float = 0, name: str = "SGD_Torch"):
        super(SGD_Torch, self).__init__(name)
        self.parameter_list = ["learning_rate", "momentum"]
        self.track_list = self.register_optimizer(object_to_track_list=object_to_track_list)
        self.optimizer = optim.SGD(params=self.track_list, lr=learning_rate, momentum=momentum)
        self.linked_loss = None

    def get_optimizer(self):
        return self.optimizer

    def link_loss_tensor(self, tensor: Tensor_Torch):
        self.linked_loss = tensor

    def register_optimizer(self, object_to_track_list: list):
        # check type for object_to_track - object_to_track (node)
        """
        optim.add_param_group({ 'params': parameters, 'lr': ...})
        """
        track_list = list()
        for object_to_track_item in object_to_track_list:
            generated_dict = dict()
            object_to_track = object_to_track_item["object"]
            if isinstance(object_to_track, LayerWrapper):
                generated_dict["params"] = object_to_track.get_layer().parameters()
            elif isinstance(object_to_track, Tensor_Torch):
                generated_dict["params"] = object_to_track.get_linked_tensor()
            for parameter in self.parameter_list:
                if parameter in object_to_track_item.keys():
                    generated_dict[parameter] = object_to_track_item[parameter]
            track_list.append(generated_dict)
        return track_list

    def step(self):
        self.optimizer.step()

    def backward(self):
        self.linked_loss.get_linked_tensor().backward()

    def clear_gradient(self):
        self.optimizer.zero_grad()

    def get_description(self):
        return "SGD Optimizer"


class LBFGS_Torch(optimizer.LBFGS):
    def __init__(self, learning_rate: float=1, name:str = "LBFGS_Torch"):
        super(LBFGS_Torch, self).__init__(name)
        self.parameter_list = ["learning_rate"]
        self.track_list = self.register_optimizer(object_to_track_list=object_to_track_list)
        self.optimizer = optim.LBFGS(params=self.track_list, lr=learning_rate)
        self.linked_loss = None

    def get_optimizer(self):
        return self.optimizer

    def link_loss_tensor(self, tensor: Tensor_Torch):
        self.linked_loss = tensor

    def register_optimizer(self, object_to_track_list: list):
        # check type for object_to_track - object_to_track (node)
        """
        optim.add_param_group({ 'params': parameters, 'lr': ...})
        """
        track_list = list()
        for object_to_track_item in object_to_track_list:
            generated_dict = dict()
            object_to_track = object_to_track_item["object"]
            if isinstance(object_to_track, LayerWrapper):
                generated_dict["params"] = object_to_track.get_layer().parameters()
            elif isinstance(object_to_track, Tensor_Torch):
                generated_dict["params"] = object_to_track.get_linked_tensor()
            for parameter in self.parameter_list:
                if parameter in object_to_track_item.keys():
                    generated_dict[parameter] = object_to_track_item[parameter]
            track_list.append(generated_dict)
        return track_list

    def step(self):
        self.optimizer.step()

    def backward(self):
        self.linked_loss.get_linked_tensor().backward()

    def clear_gradient(self):
        self.optimizer.zero_grad()

    def get_description(self):
        return "LBFGS Optimizer"