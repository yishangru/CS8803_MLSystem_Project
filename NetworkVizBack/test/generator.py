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
inputAPITorchString = {
    InputType.ImageLoader: "input_Torch.ImageLoader_Torch",
    InputType.RandomLoader: "input_Torch.RandomLoader_Torch",
    InputType.ConstantLoader: "input_Torch.ConstantLoader_Torch",
    InputType.TensorLoader: "input_Torch.TensorLoader_Torch",
    ImageDataSetType.MnistLoader: "input_Torch.MnistDataSetLoader_Torch"
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
constantAPITorchString = {
    ConstantType.ImageConstant: "input_Torch.ImageConstant_Torch",
    ConstantType.RandomConstant: "input_Torch.RandomConstant_Torch",
    ConstantType.ConstantConstant: "input_Torch.ConstantConstant_Torch",
    ConstantType.TensorConstant: "input_Torch.TensorConstant_Torch"
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
layerAPITorchString = {
    LayerType.ReLU: "layer_Torch.ReLU_Torch",
    LayerType.Linear: "layer_Torch.Linear_Torch",
    LayerType.Conv2d: "layer_Torch.Conv2d_Torch",
    LayerType.MaxPool2d: "layer_Torch.MaxPool2d_Torch",
    LayerType.BatchNorm2d: "layer_Torch.BatchNorm2d_Torch",
    LayerType.LogSoftMax: "layer_Torch.LogSoftmax_Torch",
    LayerType.MSELoss: "layer_Torch.MSELoss_Torch",
    LayerType.NLLLoss: "layer_Torch.NLLLoss_Torch"
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
transformAPITorchString = {
    TransformType.FlatTransform: "transform_Torch.FlatTransform_Torch",
    TransformType.NormalizeTransform: "transform_Torch.NormalizeTransform_Torch",
    TransformType.DataClampTransform: "transform_Torch.DataClampTransform_Torch",
    TransformType.DetachTransform: "transform_Torch.DetachTransform_Torch",
    TransformType.AddTransform: "transform_Torch.AddTransform_Torch",
    TransformType.GetGramMatrix: "transform_Torch.GetGramMatrix_Torch"
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
optimizerAPITorchString = {
    OptimizerType.SGD: "optimizer_Torch.SGD_Torch",
    OptimizerType.LBFGS: "optimizer_Torch.LBFGS_Torch"
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
monitorAPITorchString = {
    MonitorType.MonitorSaver: "monitor_Torch.MonitorSaver_Torch",
    MonitorType.MonitorFinal: "monitor_Torch.MonitorFinal_Torch"
}
monitorPortTorch = {
    MonitorType.MonitorSaver: [1],
    MonitorType.MonitorFinal: []
}


# API Mapping
inputAPI = {
    "PyTorch": inputAPITorch
}
inputPort = {
    "PyTorch": inputPortTorch
}
inputExcludeParameter = {
    "PyTorch": {"self", "name", "device"}
}

constantAPI = {
    "PyTorch": constantAPITorch
}
constantPort = {
    "PyTorch": constantPortTorch
}
constantExcludeParameter = {
    "PyTorch": {"self", "name", "device"}
}

layerAPI = {
    "PyTorch": layerAPITorch
}
layerPort = {
    "PyTorch": layerPortTorch
}
layerExcludeParameter = {
    "PyTorch": {"self", "name", "device", "import_layer"}
}

transformAPI = {
    "PyTorch": transformAPITorch
}
transformPort = {
    "PyTorch": transformPortTorch
}
transformExcludeParameter = {
    "PyTorch": {"self", "name"}
}

optimizerAPI = {
    "PyTorch": optimizerAPITorch
}
optimizerPort = {
    "PyTorch": optimizerPortTorch
}
optimizerExcludeParameter = {
    "PyTorch": {"self", "name", "object_to_track_list"}
}

monitorAPI = {
    "PyTorch": monitorAPITorch
}
monitorPort = {
    "PyTorch": monitorPortTorch
}
monitorExcludeParameter = {
    "PyTorch": {"self", "name"}
}

# general mapping for type
generalTypeMapping = {
    "input": inputTypeMapping,
    "constant": constantTypeMapping,
    "layer": layerTypeMapping,
    "transform": transformTypeMapping,
    "optimizer": optimizerTypeMapping,
    "monitor": monitorTypeMapping
}
generalTypeReverseMapping = {
    "input": inputTypeReverseMapping,
    "constant": constantTypeReverseMapping,
    "layer": layerTypeReverseMapping,
    "transform": transformTypeReverseMapping,
    "optimizer": optimizerTypeReverseMapping,
    "monitor": monitorTypeReverseMapping
}
generalAPIMapping = {
    "input": inputAPI,
    "constant": constantAPI,
    "layer": layerAPI,
    "transform": transformAPI,
    "optimizer": optimizerAPI,
    "monitor": monitorAPI
}
generalExcludeMapping = {
    "input": inputExcludeParameter,
    "constant": constantExcludeParameter,
    "layer": layerExcludeParameter,
    "transform": transformExcludeParameter,
    "optimizer": optimizerExcludeParameter,
    "monitor": monitorExcludeParameter
}
generalPortMapping = {
    "input": inputPort,
    "constant": constantPort,
    "layer": layerPort,
    "transform": transformPort,
    "optimizer": optimizerPort,
    "monitor": monitorPort
}

# default value for json
defaultParameterMapping = {
    "int": -1,
    "list": "[]",
    "tuple": "()",
    "str": "",
    "bool": -1,
    "float": -1
}

def generateAPI(API):
    import json
    import inspect
    passNodeList = list()
    for MetaNodeType in generalTypeMapping.keys():
        # node to request
        NodeRequest = generalTypeMapping[MetaNodeType]
        # api to request
        APIRequest = generalAPIMapping[MetaNodeType][API]
        # para to exclude
        ParaExclude = generalExcludeMapping[MetaNodeType][API]
        # port to request
        PORTRequest = generalPortMapping[MetaNodeType][API]
        # generate for all node type in the meta node type
        for NodeType in NodeRequest.keys():
            RequestNode = NodeRequest[NodeType]
            node = {"node": NodeType,
                    "type": MetaNodeType,
                    "api": API,
                    "ports": PORTRequest[RequestNode],
                    "source": "",
                    "description": APIRequest[RequestNode].get_description(),
                    "parameters": list()}
            signature = inspect.signature(APIRequest[RequestNode].__init__)
            for param in signature.parameters.values():
                if param.name not in ParaExclude:
                    paraDict = {"ParaName": param.name, "ParaClass": param.annotation.__name__}
                    if param.default is param.empty:
                        paraDict["ParaValue"] = defaultParameterMapping[param.annotation.__name__]
                        paraDict["Required"] = 1
                    else:
                        paraDict["ParaValue"] = param.default
                        paraDict["Required"] = 0
                    node["parameters"].append(paraDict)
            passNodeList.append(node)
    writeFile = open("./VizAPI.json", mode="w", encoding="utf-8")
    json.dump(passNodeList, fp=writeFile, indent=2)
    writeFile.close()

#generateAPI("PyTorch")

# ------------------- json parse for model generation ------------------- #
"""
format -
    nodes: [
        {
            id:
            node:
            type:
            api:
            source:
            parameters: [
                {
                    paraName:
                    paraClass
                    paraValue:
                }
            ]
        }
    ]
    links: [
        {
            start: id
            portStart: int
            end: id
            portEnd: int
        }
    ]
    block: [
        {   
            name:
            id: []
        }
    ]
"""

import ast
import json
import collections
import viz_api.viz_pytorch_api.node as VizNode_Torch

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

inputNodeAPIString = {
    "PyTorch": "VizNode_Torch.InputNode_Torch"
}
constantNodeAPIString = {
    "PyTorch": "VizNode_Torch.ConstantNode_Torch"
}
layerNodeAPIString = {
    "PyTorch": "VizNode_Torch.LayerNode_Torch"
}
transformNodeAPIString = {
    "PyTorch": "VizNode_Torch.TransformNode_Torch"
}
generalNodeMappingString = {
    "input": inputNodeAPIString,
    "constant": constantNodeAPIString,
    "layer": layerNodeAPIString,
    "transform": transformNodeAPIString,
}

inputAPIString = {
    "PyTorch": inputAPITorchString
}
constantAPIString = {
    "PyTorch": constantAPITorchString
}
layerAPIString = {
    "PyTorch": layerAPITorchString
}
transformAPIString = {
    "PyTorch": transformAPITorchString
}
optimizerAPIString = {
    "PyTorch": optimizerAPITorchString
}
monitorAPIString = {
    "PyTorch": monitorAPITorchString
}

generalAPIMappingString = {
    "input": inputAPIString,
    "constant": constantAPIString,
    "layer": layerAPIString,
    "transform": transformAPIString,
    "optimizer": optimizerAPIString,
    "monitor": monitorAPIString
}

# part 1 generate the requirements
RequirementHeader = {
    "PyTorch": "import os\nimport torch\nfrom viz_api.viz_pytorch_api import input as input_Torch\n"
               "from viz_api.viz_pytorch_api import layer as layer_Torch\n"
               "from viz_api.viz_pytorch_api import monitor as monitor_Torch\n"
               "from viz_api.viz_pytorch_api import transform as transform_Torch\n"
               "from viz_api.viz_pytorch_api import optimizer as optimizer_Torch\n"
               "from viz_api.viz_pytorch_api import node as VizNode_Torch\n\n"
}

# part 2 generate the requirements
def generateMonitor(monitorList):
    global nodeManager, recordDict

    generateMonitorString = "# -------------------- define model monitor (saving, device) -------------------- #"
    managerMonitor = None

    for monitor in monitorList:
        api = monitor["api"]
        nodeType = monitor["type"]
        nodeName = monitor["node"]
        typeRequest = generalTypeMapping[nodeType][nodeName]
        constructor = generalAPIMappingString[nodeType][api][typeRequest]

        parameterDict = dict()
        for param in monitor["parameters"]:
            parameterDict[param["paraName"]] = ast.literal_eval(param["paraValue"])
        generateName = nodeType + "_" + nodeName
        parameterDict["name"] = generateName + "_" + str(nodeManager.get_node_id(generateName))

        if nodeName == "Manager":
            managerMonitor = parameterDict["name"]

        generateDictName = parameterDict["name"] + "_dict"
        generateMonitorString = generateMonitorString + generateDictName + " = " + json.dumps(parameterDict) + "\n" + \
                         parameterDict["name"] + " = " + constructor + "(**" + generateDictName + ")\n\n"

        infoDict = {
            "name": parameterDict["name"],
            "type": monitor["type"],
            "node": monitor["node"]
        }
        recordDict[monitor["id"]] = infoDict

    return generateMonitorString, managerMonitor

# part 3 generate load and node
def generateNodeAndLoad(nodeList, managerMonitor):
    global nodeManager, recordDict

    generateLoadString = '# -------------------- load pretrained or system model -------------------- #\n' \
                         'loadPath = "./static/model/system"\n' \
                         'globalImportDict = dict()\n\n'\
                         'def loadLayerNode(name, source):\n'\
                         "\tif os.path.exists(os.path.join(loadPath, name)):\n"\
                         "\t\tglobalImportDict[name] = torch.load(os.path.join(loadPath, name))\n"\
                         '\telif source != "" and os.path.exists(os.path.join(loadPath, source)):\n'\
                         "\t\tglobalImportDict[name] = torch.load(os.path.join(loadPath, source))\n"\
                         "\telse:\n"\
                         "\t\tglobalImportDict[name] = None\n\n"
    generateNodeString = "# -------------------- define model node structure -------------------- #"

    for node in nodeList:
        api = node["api"]
        nodeType = node["type"]
        nodeName = node["node"]
        typeRequest = generalTypeMapping[nodeType][nodeName]
        constructor = generalAPIMappingString[nodeType][api][typeRequest]

        parameterDict = dict()
        for param in node["parameters"]:
            parameterDict[param["paraName"]] = ast.literal_eval(param["paraValue"])
        generateName = nodeType + "_" + nodeName
        parameterDict["name"] = generateName + "_" + str(nodeManager.get_node_id(generateName))

        if nodeType != "transform":
            parameterDict["device"] = managerMonitor + ".device"

        if nodeType == "layer":
            generateLoadString = generateLoadString + "loadLayerNode(" + parameterDict["name"] + "," + node["source"] + ")\n"
            parameterDict["import_layer"] = "globalImportDict[" + parameterDict["name"] + "]"

        nodeConstructor = generalNodeMappingString[nodeType][api]
        if nodeType == "input" and nodeName == "MNIST":
            nodeConstructor = "VizNode_Torch.MnistNode_Torch"

        generateDictName = parameterDict["name"] + "_dict"
        generateNodeString = generateNodeString + generateDictName + " = " + json.dumps(parameterDict) + "\n" + \
                         parameterDict["name"] + " = " + nodeConstructor + "(" + constructor + ", " + generateDictName + ")\n\n"

        infoDict = {
            "name": parameterDict["name"],
            "type": node["type"],
            "node": node["node"]
        }
        recordDict[node["id"]] = infoDict

    generateLoadString += "\n\n\n"
    generateNodeString += "\n\n\n"
    return generateLoadString, generateNodeString


def generateSystemModel(monitorList, nodeList, linkList, blockList, optimizerList):
    global nodeManager, recordDict
    recordDict = dict()
    nodeManager = GlobalManager()
    monitorString, managerMonitor = generateMonitor(monitorList)  # for the model iteration
    loadString, nodeString = generateNodeAndLoad(nodeList, managerMonitor)  # for node declare and load model

    # parse link
    

def generateOptimizer():
    pass

def generateTraining():
    pass

def generateSaving():
    pass

def generateEvaluation():
    pass

nodeName = "MNIST"
nodeType = "input"
api = "PyTorch"
typeRequest = generalTypeMapping[nodeType][nodeName]
print(generalAPIMappingString[nodeType][api][typeRequest])