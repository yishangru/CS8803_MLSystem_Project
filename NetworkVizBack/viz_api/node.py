# upper function call for corresponding layer defined in platform directory

"""
Layer nodes can have multiple outputs.
Constant nodes can have multiple outputs.
Transform nodes can have multiple outputs.
Input nodes can have multiple outputs.
"""

# for a node
class Node(object):
    def __init__(self, name: str):
        # name is the id identification
        super(Node, self).__init__()
        self.name = name

    def set_output_port(self, number: int):
        raise NotImplementedError

    def get_output_tensor(self):
        raise NotImplementedError

    def forward(self, inputParaList: list):
        raise NotImplementedError

    # return KB in memory usage for the loaded tensor
    def get_tensor_memory_size(self):
        raise NotImplementedError

    # return KB in memory usage for gradients of the loaded tensor
    def get_tensor_grad_memory_size(self):
        raise NotImplementedError


# wrapper for layer node
class LayerNode(Node):
    def __init__(self, name: str):
        super(LayerNode, self).__init__(name)

    # return KB in memory usage for layer feature (weight, bias)
    def get_layer_feature_memory_size(self):
        raise NotImplementedError

    # return KB in memory usage for layer gradients
    def get_layer_grad_memory_size(self):
        raise NotImplementedError


# wrapper for transform node
class TransformNode(Node):
    def __init__(self, name):
        super(TransformNode, self).__init__(name)


# wrapper for input node
class InputNode(Node):
    def __init__(self, name: str):
        super(InputNode, self).__init__(name)


# wrapper for constant node
class ConstantNode(Node):
    def __init__(self, name: str):
        super(ConstantNode, self).__init__(name)