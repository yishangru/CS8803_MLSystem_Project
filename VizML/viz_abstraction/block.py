from viz_api import node
"""
    For block level abstraction:
        1. Node
        2. Hardware (possible)
"""


class Block(object):
    def __init__(self, nodeList: list, name: str="Block"):  # block information
        super(Block, self).__init__()
        self.name = name
        self.nodeList = nodeList

    # return KB in memory usage for the loaded tensor
    def get_tensor_memory_size(self):
        for node in self.nodeList:


    # return KB in memory usage for gradients of the loaded tensor
    def get_tensor_grad_memory_size(self):
        raise NotImplementedError

    # return KB in memory usage for layer feature (weight, bias)
    def get_layer_feature_memory_size(self):
        raise NotImplementedError

    # return KB in memory usage for layer gradients
    def get_layer_grad_memory_size(self):
        raise NotImplementedError