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
        self.meta = self.initial_meta_profiling()

    def initial_meta_profiling(self):
        return {"tensor":
                         {"mem":
                              {"max": 0, "min": float("inf")},
                          "grad":
                              {"max": 0, "min": float("inf")}
                          },
                     "layer":
                         {"mem":
                              {"max": 0, "min": float("inf")},
                          "grad":
                              {"max": 0, "min": float("inf")}
                          }
                     }

    def get_meta_record(self):
        return self.meta

    def clear_meta_for_next_epoch(self):
        self.meta = self.initial_meta_profiling()

    def update_record(self):
        tensor_mem = self.get_tensor_memory_size()
        tensor_grad = self.get_tensor_grad_memory_size()
        layer_mem = self.get_layer_feature_memory_size()
        layer_grad = self.get_layer_grad_memory_size()

        self.meta["tensor"]["mem"]["max"] = max(self.meta["tensor"]["mem"]["max"], tensor_mem)
        self.meta["tensor"]["mem"]["min"] = min(self.meta["tensor"]["mem"]["min"], tensor_mem)
        self.meta["tensor"]["grad"]["max"] = max(self.meta["tensor"]["mem"]["max"], tensor_grad)
        self.meta["tensor"]["grad"]["min"] = min(self.meta["tensor"]["mem"]["min"], tensor_grad)

        self.meta["layer"]["mem"]["max"] = max(self.meta["layer"]["mem"]["max"], layer_mem)
        self.meta["layer"]["mem"]["min"] = min(self.meta["layer"]["mem"]["min"], layer_mem)
        self.meta["layer"]["grad"]["max"] = max(self.meta["layer"]["grad"]["max"], layer_grad)
        self.meta["layer"]["grad"]["min"] = min(self.meta["layer"]["grad"]["min"], layer_grad)

    # return KB in memory usage for the loaded tensor
    def get_tensor_memory_size(self):
        tensor_memory_size = 0
        for blockNode in self.nodeList:
            assert (isinstance(blockNode, node.Node))
            tensor_memory_size += blockNode.get_tensor_memory_size()

    # return KB in memory usage for gradients of the loaded tensor
    def get_tensor_grad_memory_size(self):
        tensor_memory_size = 0
        for blockNode in self.nodeList:
            assert (isinstance(blockNode, node.Node))
            tensor_memory_size += blockNode.get_tensor_grad_memory_size()

    # return KB in memory usage for layer feature (weight, bias)
    def get_layer_feature_memory_size(self):
        layer_feature_size = 0
        for blockNode in self.nodeList:
            if isinstance(blockNode, node.LayerNode):
                layer_feature_size += blockNode.get_layer_feature_memory_size()

    # return KB in memory usage for layer gradients
    def get_layer_grad_memory_size(self):
        layer_grad_size = 0
        for blockNode in self.nodeList:
            if isinstance(blockNode, node.LayerNode):
                layer_grad_size += blockNode.get_layer_grad_memory_size()