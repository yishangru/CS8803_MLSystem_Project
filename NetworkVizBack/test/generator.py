import collections

class GlobalManager(object):
    def __init__(self, name: str = "GlobalManager"):
        super(GlobalManager, self).__init__()
        self.name = name
        self.block_id = 0
        self.recorder = collections.defaultdict(int)

    def get_block_id(self):
        block_id_return = self.block_id
        self.block_id += 1
        return block_id_return

    def get_node_id(self, name):
        node_id_return = self.recorder[name]
        self.recorder[name] += 1
        return node_id_return

import viz_api.node as VizNode
from viz_api.tensor import DataType
from viz_api.input import InputType, ImageDataSetType, ConstantType
from viz_api.layer import LayerType
from viz_api.monitor import MonitorType
from viz_api.transform import TransformType
from viz_api.optimizer import OptimizerType

from viz_api.viz_pytorch_api import input as input_Torch
from viz_api.viz_pytorch_api import layer as layer_Torch
from viz_api.viz_pytorch_api import monitor as monitor_Torch
from viz_api.viz_pytorch_api import transform as transform_Torch
from viz_api.viz_pytorch_api import optimizer as optimizer_Torch

"""
Port:
    1: main input
    2: sub input
    3: meta
    4: main output
    5: sub output (label)
"""

# ------------------- input mapping ------------------- #
inputTypeMapping = {
    "ImageTensor": InputType.ImageLoader,
    "RandomTensor": InputType.RandomLoader,
    "ConstantTensor": InputType.ConstantLoader,
    "SavedTensor": InputType.TensorLoader,
    "MNIST": ImageDataSetType.MnistLoader
}
inputTypeReverseMapping = {v: k for k, v in inputTypeMapping.items()}

# api for pytorch
inputAPITorch = {
    InputType.ImageLoader: input_Torch.ImageLoader_Torch,
    InputType.RandomLoader: input_Torch.RandomLoader_Torch,
    InputType.ConstantLoader: input_Torch.ConstantLoader_Torch,
    InputType.TensorLoader: input_Torch.TensorLoader_Torch,
    ImageDataSetType.MnistLoader: input_Torch.MnistDataSetLoader_Torch
}
# port for pytorch
inputPortTorch = {
    InputType.ImageLoader: [3, 4],
    InputType.RandomLoader: [3, 4],
    InputType.ConstantLoader: [3, 4],
    InputType.TensorLoader: [3, 4],
    ImageDataSetType.MnistLoader: [3, 4, 5]
}


# ------------------- constant mapping ------------------- #
constantTypeMapping = {
    "ImageTensor": ConstantType.ImageConstant,
    "RandomTensor": ConstantType.RandomConstant,
    "ConstantTensor": ConstantType.ConstantConstant,
    "SavedTensor": ConstantType.TensorConstant
}
constantTypeReverseMapping = {v: k for k, v in constantTypeMapping.items()}

# api for pytorch
constantAPITorch = {
    ConstantType.ImageConstant: input_Torch.ImageConstant_Torch,
    ConstantType.RandomConstant: input_Torch.RandomConstant_Torch,
    ConstantType.ConstantConstant: input_Torch.ConstantConstant_Torch,
    ConstantType.TensorConstant: input_Torch.TensorConstant_Torch
}
# port for pytorch
constantPortTorch = {
    ConstantType.ImageConstant: [3, 4],
    ConstantType.RandomConstant: [3, 4],
    ConstantType.ConstantConstant: [3, 4],
    ConstantType.TensorConstant: [3, 4]
}


# ------------------- layer mapping ------------------- #
layerTypeMapping = {
    "ReLU": LayerType.ReLU,
    "Linear": LayerType.Linear,
    "Conv2D": LayerType.Conv2d,
    "MaxPool2D": LayerType.MaxPool2d,
    "BatchNorm2D": LayerType.BatchNorm2d,
    "LogSoftMax": LayerType.LogSoftMax,
    "MSELoss": LayerType.MSELoss,
    "NLLLoss": LayerType.NLLLoss
}
layerTypeReverseMapping = {v: k for k, v in layerTypeMapping.items()}

# api for pytorch
layerAPITorch = {
    LayerType.ReLU: layer_Torch.ReLU_Torch,
    LayerType.Linear: layer_Torch.Linear_Torch,
    LayerType.Conv2d: layer_Torch.Conv2d_Torch,
    LayerType.MaxPool2d: layer_Torch.MaxPool2d_Torch,
    LayerType.BatchNorm2d: layer_Torch.BatchNorm2d_Torch,
    LayerType.LogSoftMax: layer_Torch.LogSoftmax_Torch,
    LayerType.MSELoss: layer_Torch.MSELoss_Torch,
    LayerType.NLLLoss: layer_Torch.NLLLoss_Torch
}
# port for pytorch
layerPortTorch = {
    LayerType.ReLU: [1, 3, 4],
    LayerType.Linear: [1, 3, 4],
    LayerType.Conv2d: [1, 3, 4],
    LayerType.MaxPool2d: [1, 3, 4],
    LayerType.BatchNorm2d: [1, 3, 4],
    LayerType.LogSoftMax: [1, 3, 4],
    LayerType.MSELoss: [1, 3, 4],
    LayerType.NLLLoss: [1, 3, 4]
}


# ------------------- transform mapping ------------------- #
transformTypeMapping = {
    "Flatten": TransformType.FlatTransform,
    "Normalize": TransformType.NormalizeTransform,
    "ClampData": TransformType.DataClampTransform,
    "Detach": TransformType.DetachTransform,
    "Adder": TransformType.AddTransform,
    "GetGramMatrix": TransformType.GetGramMatrix
}
transformTypeReverseMapping = {v: k for k, v in transformTypeMapping.items()}

# api for pytorch
transformAPITorch = {
    TransformType.FlatTransform: transform_Torch.FlatTransform_Torch,
    TransformType.NormalizeTransform: transform_Torch.NormalizeTransform_Torch,
    TransformType.DataClampTransform: transform_Torch.DataClampTransform_Torch,
    TransformType.DetachTransform: transform_Torch.DetachTransform_Torch,
    TransformType.AddTransform: transform_Torch.AddTransform_Torch,
    TransformType.GetGramMatrix: transform_Torch.GetGramMatrix_Torch
}
# port for pytorch
transformPortTorch = {
    TransformType.FlatTransform: [1, 3, 4],
    TransformType.NormalizeTransform: [1, 3, 4],
    TransformType.DataClampTransform: [1, 3, 4],
    TransformType.DetachTransform: [1, 3, 4],
    TransformType.AddTransform: [1, 3, 4],
    TransformType.GetGramMatrix: [1, 3, 4]
}

# ------------------- optimize mapping ------------------- #
optimizerTypeMapping = {
    "SGD": OptimizerType.SGD,
    "LBFGS": OptimizerType.LBFGS
}
optimizerTypeReverseMapping = {v: k for k, v in optimizerTypeMapping.items()}

# api for pytorch
optimizerAPITorch = {
    OptimizerType.SGD: optimizer_Torch.SGD_Torch,
    OptimizerType.LBFGS: optimizer_Torch.LBFGS_Torch
}
# port for pytorch
optimizerPortTorch = {
    OptimizerType.SGD: [1, 2],
    OptimizerType.LBFGS: [1, 2]
}

# ------------------- monitor mapping ------------------- #
monitorTypeMapping = {
    "Saver": MonitorType.MonitorSaver,
    "Manager": MonitorType.MonitorFinal
}
monitorTypeReverseMapping = {v: k for k, v in monitorTypeMapping.items()}

# api for pytorch
monitorAPITorch = {
    MonitorType.MonitorSaver: monitor_Torch.MonitorSaver_Torch,
    MonitorType.MonitorFinal: monitor_Torch.MonitorFinal_Torch
}
monitorPortTorch = {
    MonitorType.MonitorSaver: [1],
    MonitorType.MonitorFinal: []
}



# API Mapping
generalMapping = {
    "input": inputTypeMapping,
    "constant": constantTypeMapping,
    "layer": layerTypeMapping,
    "transform": transformTypeMapping,
    "optimizer": optimizerTypeMapping,
    "monitor": monitorTypeMapping
}

inputAPI = {
    "PyTorch": inputAPITorch
}
inputPort = {
    "PyTorch": inputPortTorch
}
inputExcludeParameter = {
    "Pytorch": {"self", "name", "device"}
}

constantAPI = {
    "PyTorch": constantAPITorch
}
constantPort = {
    "PyTorch": constantPortTorch
}
constantExcludeParameter = {
    "Pytorch": {"self", "name", "device"}
}

layerAPI = {
    "PyTorch": layerAPITorch
}
layerPort = {
    "PyTorch": layerPortTorch
}
layerExcludeParameter = {
    "Pytorch": {"self", "name", "device"}
}

transformAPI = {
    "PyTorch": transformAPITorch
}
transformPort = {
    "PyTorch": transformPortTorch
}
transformExcludeParameter = {
    "Pytorch": {"self", "name", "device"}
}

optimizerAPI = {
    "PyTorch": optimizerAPITorch
}
optimizerPort = {
    "PyTorch": optimizerPortTorch
}
optimizerExcludeParameter = {
    "Pytorch": {"self", "name", "device"}
}

monitorAPI = {
    "PyTorch": monitorAPITorch
}
monitorPort = {
    "PyTorch": monitorPortTorch
}
monitorExcludeParameter = {
    "Pytorch": {"self", "name", "device"}
}

defaultParameterMapping = {
    "int": -1,
    "list": "[]",
    "tuple": "()",
    "str": "",
    "bool": -1
}

def generateAPIFormat(API):
    import inspect
    passDict = list()
    for node in inputAPI[API].keys():
        node = {"name": "LogSoftmax", "type": "Layer", "api": "torch", "description": ,
                "parameters": list()}
        signature = inspect.signature(inputAPI[API][node].__init__)
        for param in signature.parameters.values():
            if param.name not in inputExcludeParameter:
                paraDict = {"ParaName": param.name, "ParaClass": param.annotation.__name__}
                if param.default is param.empty:
                    paraDict["ParaValue"] = defaultParameterMapping[param.annotation.__name__]
                    paraDict["Required"] = 1
                else:
                    paraDict["ParaValue"] = param.default
                    paraDict["Required"] = 0
                node["Parameters"].append(paraDict)



def generateSystemModel(API, nodeList, linkList):
    recordDict = dict()
    nodeManager = GlobalManager()
    for node in nodeList:
        """
        format -
            {
                id:
                NodeType:
                type:
                source:
                params: [
                    {
                        paraName:
                        paraValue:
                    }
                ]
            }
        """