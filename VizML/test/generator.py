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
    LayerType.MSELoss: [1, 2, 3, 4],
    LayerType.NLLLoss: [1, 2, 3, 4]
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
    TransformType.FlatTransform: [1, 4],
    TransformType.NormalizeTransform: [1, 4],
    TransformType.DataClampTransform: [1, 4],
    TransformType.DetachTransform: [1, 4],
    TransformType.AddTransform: [1, 4],
    TransformType.GetGramMatrix: [1, 4]
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
    "int": "-1",
    "list": "[]",
    "tuple": "()",
    "str": "",
    "bool": "False",
    "float": "-1"
}

def generateAPI(API, GeneratePath):
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
                    "parameters": dict()}
            signature = inspect.signature(APIRequest[RequestNode].__init__)
            for param in signature.parameters.values():
                if param.name not in ParaExclude:
                    paraDict = {"ParaName": param.name, "ParaClass": param.annotation.__name__}
                    if param.default is param.empty:
                        paraDict["ParaValue"] = defaultParameterMapping[param.annotation.__name__]
                        paraDict["Required"] = 1
                    else:
                        paraDict["ParaValue"] = str(param.default)
                        paraDict["Required"] = 0
                    node["parameters"][paraDict["ParaName"]] = paraDict
            passNodeList.append(node)
    writeFile = open(GeneratePath, mode="w+", encoding="utf-8")
    json.dump(passNodeList, fp=writeFile, indent=2)
    writeFile.close()

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
    block: [
        {   
            id:
            name:
            ids: []
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
"""

import ast
import json
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
    "General": "import os\n"
               "import collections\n"
               "from viz_abstraction.block import Block\n"
               "from model_profiling.profiler import BlockProfiler\n",
    "PyTorch": "import torch\nfrom viz_api.viz_pytorch_api import input as input_Torch\n"
               "from viz_api.viz_pytorch_api import layer as layer_Torch\n"
               "from viz_api.viz_pytorch_api import monitor as monitor_Torch\n"
               "from viz_api.viz_pytorch_api import transform as transform_Torch\n"
               "from viz_api.viz_pytorch_api import optimizer as optimizer_Torch\n"
               "from viz_api.viz_pytorch_api import node as VizNode_Torch\n"
}

# part 2 generate the requirements
def generateMonitor(monitorList):
    global nodeManager, recordDict

    generateMonitorString = "# -------------------- define model monitor (mode, saving, device) -------------------- #\n" \
                            "model_running_train = True\n\n"
    managerMonitor = None

    for monitor in monitorList:
        api = monitor["api"]
        nodeType = monitor["type"]
        nodeName = monitor["node"]
        typeRequest = generalTypeMapping[nodeType][nodeName]
        constructor = generalAPIMappingString[nodeType][api][typeRequest]

        parameterDict = dict()
        for param in monitor["parameters"]:
            if param["ParaClass"] == "str":
                parameterDict[param["ParaName"]] = param["ParaValue"]
            else:
                parameterDict[param["ParaName"]] = ast.literal_eval(param["ParaValue"])
        generateName = nodeType + "_" + nodeName
        parameterDict["name"] = generateName + "_" + str(nodeManager.get_node_id(generateName))

        if nodeName == "Manager":
            managerMonitor = parameterDict["name"]

        generateDictName = parameterDict["name"] + "_dict"
        generateMonitorString = generateMonitorString + generateDictName + " = " + str(parameterDict) + "\n" + \
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
                         'loadPath = "../system"\n' \
                         'globalImportDict = dict()\n\n'\
                         'def loadLayerNode(name, source):\n'\
                         "\tif os.path.exists(os.path.join(loadPath, name)):\n"\
                         "\t\tglobalImportDict[name] = torch.load(os.path.join(loadPath, name))\n"\
                         '\telif source != "" and os.path.exists(os.path.join(loadPath, source)):\n'\
                         "\t\tglobalImportDict[name] = torch.load(os.path.join(loadPath, source))\n"\
                         "\telse:\n"\
                         "\t\tglobalImportDict[name] = None\n\n"
    generateNodeString = "# -------------------- define model node structure -------------------- #\n"
    generateModeString = "if not model_running_train:\n"

    for node in nodeList:
        api = node["api"]
        nodeType = node["type"]
        nodeName = node["node"]
        typeRequest = generalTypeMapping[nodeType][nodeName]
        constructor = generalAPIMappingString[nodeType][api][typeRequest]

        parameterDict = dict()
        for param in node["parameters"]:
            if param["ParaClass"] == "str":
                parameterDict[param["ParaName"]] = param["ParaValue"]
            else:
                parameterDict[param["ParaName"]] = ast.literal_eval(param["ParaValue"])
        generateName = nodeType + "_" + nodeName
        parameterDict["name"] = generateName + "_" + str(nodeManager.get_node_id(generateName))

        parameterDictString = str(parameterDict)
        if nodeType != "transform":
            parameterDictString = parameterDictString.rstrip("}") + ", "
            parameterDictString += ("'device': " + managerMonitor + ".device" + "}")

        if nodeType == "layer":
            parameterDictString = parameterDictString.rstrip("}") + ", "
            parameterDictString += ("'import_layer': globalImportDict['" + parameterDict["name"] + "']}")
            generateLoadString = generateLoadString + "loadLayerNode('" + parameterDict["name"] + "', '" + node["source"] + "')\n"

        nodeConstructor = generalNodeMappingString[nodeType][api]
        if nodeType == "input" and nodeName == "MNIST":
            nodeConstructor = "VizNode_Torch.MnistNode_Torch"

        generateDictName = parameterDict["name"] + "_dict"
        generateNodeString = generateNodeString + generateDictName + " = " + parameterDictString + "\n" + \
                         parameterDict["name"] + " = " + nodeConstructor + "(" + constructor + ", " + generateDictName + ")\n\n"

        if nodeType == "layer":
            # check mode of training
            generateModeString = generateModeString + "\t" + parameterDict["name"] + ".set_as_eval()\n"

        infoDict = {
            "name": parameterDict["name"],
            "type": node["type"],
            "node": node["node"]
        }
        recordDict[node["id"]] = infoDict

    generateLoadString += "\n\n\n"
    generateNodeString += (generateModeString + "\n\n")
    return generateLoadString, generateNodeString

# part 4 generate block recorder
def generateBlock(blockList):
    global nodeManager, recordDict

    generateBlockInitial = ["# -------------------- model block initialize -------------------- #"]
    generateBlockInitial.append("block_profiling_dict = collections.defaultdict(lambda: list())")
    generateBlockProfile = {"update": ["# -------------------- block profile -------------------- #"],
                            "record": ["# -------------------- block epoch record -------------------- #"],
                            "plot": ["# -------------------- block epoch plot -------------------- #"]}

    for block in blockList:
        blockName = "Block" + str(block["id"])
        blockNodeListName = blockName + "_nodes"
        nodeListString = ""
        for node in block["nodeIDs"]:
            nodeListString += (recordDict[node]["name"] + ", ")
        nodeListString = blockNodeListName + " = [" + nodeListString + "]"
        blockString = blockName + " = Block(" + blockNodeListName + ", '" + blockName + "')"
        generateBlockInitial.append(nodeListString)
        generateBlockInitial.append(blockString)
        generateBlockProfile["update"].append(blockName + ".update_record()")
        generateBlockProfile["record"].append("block_profiling_dict['" + blockName + "'].append(" + blockName + ".get_meta_record())")
        generateBlockProfile["record"].append(blockName + ".clear_meta_for_next_epoch()")

    if len(blockList) > 0:
        generateBlockProfile["plot"].append("model_block_profiler = BlockProfiler('./')")
        generateBlockProfile["plot"].append("model_block_profiler.generateBlockImage(block_profiling_dict)")
    else:
        generateBlockProfile["plot"].append("# no block generated, no plots")

    return generateBlockInitial, generateBlockProfile

# part 5 generate link for running logic and saving code
def generateTraining(linkList, optimizerList):
    global nodeManager, recordDict

    generateRunInitial = ["# -------------------- model forwarding initialize -------------------- #"]
    generateRunString = ["# -------------------- model running -------------------- #"]
    generateOptimizerInitial = ["# -------------------- optimize initial -------------------- #"]
    generateOptimizerString = {"clear": ["# -------------------- model optimize -------------------- #",
                                         "if model_running_train:"],
                               "optimize": ["# -------------------- model optimize -------------------- #",
                                            "if model_running_train:"]}
    generateSaveString = ["# -------------------- output save -------------------- #"]

    for optimizer in optimizerList:
        # generate name mapping
        nodeType = optimizer["type"]
        nodeName = optimizer["node"]

        generateName = nodeType + "_" + nodeName
        nodeGeneratedName = generateName + "_" + str(nodeManager.get_node_id(generateName))
        infoDict = {
            "name": nodeGeneratedName,
            "type": nodeType,
            "node": nodeName,
            "tracking": list(),
            "loss": None
        }
        recordDict[optimizer["id"]] = infoDict

    # node - port - node
    GraphDict = dict()
    VariableDict = dict()

    SaveLinkList = list()
    OptimizerLinkList = list()
    for link in linkList:
        sourceNode = link["source"]["nodeID"]
        sourcePort = link["source"]["port"]
        targetNode = link["target"]["nodeID"]
        targetPort = link["target"]["port"]
        if recordDict[targetNode]["type"] == "optimizer":
            OptimizerLinkList.append(link)
        elif recordDict[targetNode]["type"] == "monitor" and recordDict[targetNode]["node"] == "Saver":
            SaveLinkList.append(link)
        else:
            if sourceNode not in GraphDict.keys():
                GraphDict[sourceNode] = {"input": set(), "output": dict()}
            if targetNode not in GraphDict.keys():
                GraphDict[targetNode] = {"input": set(), "output": dict()}

            # set the variable link, since link is the unique
            if sourcePort != 3: # the meta port, should not present here
                linkedVariable = recordDict[sourceNode]["name"] + "_" + str(sourcePort) + "_" + recordDict[targetNode]["name"] + "_" + str(targetPort)
                if sourcePort not in GraphDict[sourceNode]["output"].keys():
                    GraphDict[sourceNode]["output"][sourcePort] = set()
                GraphDict[sourceNode]["output"][sourcePort].add(linkedVariable)
                GraphDict[targetNode]["input"].add(linkedVariable)
                VariableDict[linkedVariable] = {"sourceNode": sourceNode, "sourcePort": sourcePort,
                                                "targetNode": targetNode, "targetPort": targetPort}
            else:
                print("Meta Port in graph! Error! Should not be here !")

    # generate output copy number
    for node in GraphDict.keys():
        largestCopy = 1 # consider the possible saver and optimizer
        for outputPort in GraphDict[node]["output"]:
            largestCopy = max(len(GraphDict[node]["output"][outputPort]), largestCopy)

        generateRunInitial.append(recordDict[node]["name"] + ".set_output_port(" + str(largestCopy) + ")")
        for outputPort in GraphDict[node]["output"]:
            assignNumber = 0
            for linkedVariable in GraphDict[node]["output"][outputPort]:
                VariableDict[linkedVariable]["assignNumber"] = assignNumber
                assignNumber += 1

    # generate optimizer string
    for optimizeLink in OptimizerLinkList: # just implement for port 3
        sourceNode = optimizeLink["source"]["nodeID"]
        sourcePort = optimizeLink["source"]["port"]
        targetNode = optimizeLink["target"]["nodeID"]
        targetPort = optimizeLink["target"]["port"]
        if targetPort == 1:
            if sourcePort == 3:
                if recordDict[sourceNode]["type"] == "input" or recordDict[sourceNode]["type"] == "layer":
                    recordDict[targetNode]["tracking"].append(sourceNode)
                else:
                    print("Not implement yet for non-input and layer - stop generation")
                    return
            else:
                print("Get non meta optimize - stop generation")
                return
        elif targetPort == 2:
            if sourcePort == 4:
                linkedVariableName = recordDict[sourceNode]["name"] + "_" + str(sourcePort) + "_" + recordDict[targetNode]["name"] + "_" + str(targetPort)
                # update clean
                generateOptimizerString["clear"].append("\t" + recordDict[targetNode]["name"] + ".clear_gradient()")
                # update optimize
                generateOptimizerString["optimize"].append("\t" + linkedVariableName + " = " + recordDict[sourceNode]["name"] +
                                                           ".get_output_tensor_single(0)")
                generateOptimizerString["optimize"].append("\t" + recordDict[targetNode]["name"] + ".link_loss_tensor(" + linkedVariableName + ")")
                generateOptimizerString["optimize"].append("\t" + recordDict[targetNode]["name"] + ".backward()")
                generateOptimizerString["optimize"].append("\t" + recordDict[targetNode]["name"] + ".step()")
            else:
                print("Get unexpected loss input - stop generation")
                return

    print("\n".join(generateOptimizerString["clear"]))
    print("\n".join(generateOptimizerString["optimize"]))

    # initialize optimizer
    for optimizer in optimizerList:
        optimizerID = optimizer["id"]
        trackListName = recordDict[optimizerID]["name"] + "_track_list"
        if (len(recordDict[optimizerID]) > 0):
            trackingList = ""
            for trackNode in recordDict[optimizerID]["tracking"]:
                if recordDict[trackNode]["type"] == "input":
                    trackingList += ("{'object':" + recordDict[trackNode]["name"] + ".get_linked_layer()}, ")
                elif recordDict[trackNode]["type"] == "layer":
                    trackingList += ("{'object':" + recordDict[trackNode]["name"] + ".get_linked_layer()}, ")
                else:
                    print("Not implement yet for non-input and layer - stop generation")
                    return
            trackingList = trackListName + " = [" + trackingList + "]"
            generateOptimizerInitial.append(trackingList)

            api = optimizer["api"]
            nodeType = optimizer["type"]
            nodeName = optimizer["node"]
            typeRequest = generalTypeMapping[nodeType][nodeName]
            constructor = generalAPIMappingString[nodeType][api][typeRequest]

            parameterDict = dict()
            parameterDict["name"] = recordDict[optimizerID]["name"]
            for param in optimizer["parameters"]:
                if param["ParaClass"] == "str":
                    parameterDict[param["ParaName"]] = param["ParaValue"]
                else:
                    parameterDict[param["ParaName"]] = ast.literal_eval(param["ParaValue"])

            parameterDictString = str(parameterDict)
            parameterDictString = parameterDictString.rstrip("}") + ", "
            parameterDictString += ("'object_to_track_list': " + trackListName + "}")

            generateDictName = parameterDict["name"] + "_dict"
            generateOptimizerInitial.append(generateDictName + " = " + parameterDictString)
            generateOptimizerInitial.append(parameterDict["name"] + " = " + constructor + "(**" + generateDictName + ")")
        else:
            print("Optimizer not tracking object - stop generation")
            return
    print("\n".join(generateOptimizerInitial))

    # generate model running string - start from input
    startNode = -1
    nodeVisited = set()
    nodeToVisit = list()
    generatedOutput = set()
    for node in GraphDict.keys():
        if recordDict[node]["type"] == "input":
            if startNode == -1:
                startNode = node
            else:
                print("Multiple input - stop generation")
                return
        elif recordDict[node]["type"] == "constant":
            nodeToVisit.append(node)
    if startNode != -1:
        # start from input node, add constant node and input node to the visitlist
        nodeToVisit.insert(0, startNode)
        while len(nodeToVisit) > 0:
            presentNode = nodeToVisit.pop(0)
            if presentNode not in nodeVisited:
                if recordDict[presentNode]["type"] == "input":
                    generateRunString.append(recordDict[presentNode]["name"] + ".forward([iteration])\n")
                else:
                    mainInput = ""
                    inputVariableString = ""
                    for linkedVariable in GraphDict[presentNode]["input"]:
                        if VariableDict[linkedVariable]["targetPort"] == 1:
                            mainInput = linkedVariable
                        else:
                            inputVariableString += (", " + linkedVariable)
                    inputVariableString = mainInput + inputVariableString
                    generateRunString.append(recordDict[presentNode]["name"] + ".forward([" + inputVariableString + "])\n")

                for outputPort in GraphDict[presentNode]["output"]:
                    for linkedVariable in GraphDict[presentNode]["output"][outputPort]:
                        # variable is generated
                        generatedOutput.add(linkedVariable)
                        if outputPort == 4:
                            generateRunString.append(linkedVariable + " = " + recordDict[presentNode]["name"] +
                                        ".get_output_tensor_single(" + str(VariableDict[linkedVariable]["assignNumber"]) + ")")
                        elif outputPort == 5:
                            generateRunString.append(linkedVariable + " = " + recordDict[presentNode]["name"] +
                                        ".get_output_label_single(" + str(VariableDict[linkedVariable]["assignNumber"]) + ")")
                        # check if there is new node can be added to visit
                        nodeCandidate = VariableDict[linkedVariable]["targetNode"]
                        # check whether all input satisfy
                        whetherAdd = True
                        for linkedVariable in GraphDict[nodeCandidate]["input"]:
                            if linkedVariable not in generatedOutput:
                                whetherAdd = False
                                break
                        if whetherAdd:
                            nodeToVisit.insert(0, nodeCandidate)
                nodeVisited.add(presentNode)
    else:
        print("Can't find input node - stop generation")
        return

    print("\n".join(generateRunInitial))
    print("\n".join(generateRunString))

    # generate save string - not save overall model yet
    for saveLink in SaveLinkList:
        sourceNode = saveLink["source"]["nodeID"]
        sourcePort = saveLink["source"]["port"]
        targetNode = saveLink["target"]["nodeID"]
        targetPort = saveLink["target"]["port"]

        linkedVariableName = recordDict[sourceNode]["name"] + "_" + str(sourcePort) + "_" + recordDict[targetNode][
            "name"] + "_" + str(targetPort)
        if recordDict[sourceNode]["type"] == "input":
            generateSaveString.append(linkedVariableName + " = " + recordDict[sourceNode]["name"] + ".get_linked_input()")
        elif recordDict[sourceNode]["type"] == "constant":
            generateSaveString.append(linkedVariableName + " = " + recordDict[sourceNode]["name"] + ".get_linked_constant()")
        elif recordDict[sourceNode]["type"] == "transform" or recordDict[sourceNode]["layer"]:
            # save the first output tensor
            generateSaveString.append(linkedVariableName + " = " + recordDict[sourceNode]["name"] + ".get_output_tensor_single(0)")
        generateSaveString.append(recordDict[targetNode]["name"] + ".save_output(" + linkedVariableName + ")")

    print("\n".join(generateSaveString))

    return generateRunInitial, generateRunString, generateOptimizerInitial, generateOptimizerString, generateSaveString, startNode

# block and node id are sharing, globally unique
def generateModel(model, path):
    global nodeManager, recordDict
    recordDict = dict()
    nodeManager = GlobalManager()

    nodeList = []
    monitorList = []
    optimizerList = []

    for node in model["nodes"]:
        if node["type"] == "monitor":
            monitorList.append(node)
        elif node["type"] == "optimizer":
            optimizerList.append(node)
        else:
            nodeList.append(node)

    header = RequirementHeader["General"]
    platform = RequirementHeader["PyTorch"]
    monitorString, managerMonitor = generateMonitor(monitorList)
    loadString, nodeString = generateNodeAndLoad(nodeList, managerMonitor)  # for node declare and load model
    generateRunInitial, generateRunString, generateOptimizerInitial, generateOptimizerString, generateSaveString, startNode = generateTraining(model_json["links"], optimizerList)
    generateBlockInitial, generateBlockProfile = generateBlock(model_json["blocks"])

    # generate the overall model
    outputFile = open(path, mode="w", encoding="utf-8")
    outputFile.write(header)
    outputFile.write(platform)
    outputFile.write(monitorString)
    outputFile.write(loadString)
    outputFile.write(nodeString)
    outputFile.write("\n".join(generateRunInitial) + "\n\n")
    outputFile.write("\n".join(generateOptimizerInitial) + "\n\n")
    outputFile.write("\n".join(generateBlockInitial) + "\n\n")

    # add epoch for model running and optimizing
    model_running_string = "for model_running_epoch in range(" + managerMonitor + ".epochs):\n" + \
                           "\tfor iteration in range(" + recordDict[startNode]["name"] + ".get_linked_input().get_number_batch()):\n" + \
                           "\n".join(["\t\t" + optimizing_line for optimizing_line in generateOptimizerString["clear"]]) + "\n\n" + \
                           "\n".join(["\t\t" + block_line for block_line in generateBlockProfile["update"]]) + "\n\n" + \
                           "\n".join(["\t\t" + running_line for running_line in generateRunString]) + "\n\n" + \
                           "\n".join(["\t\t" + optimizing_line for optimizing_line in generateOptimizerString["optimize"]]) + "\n\n" + \
                           "\n".join(["\t\t" + block_line for block_line in generateBlockProfile["update"]]) + "\n\n" + \
                           "\n".join(["\t" + block_line for block_line in generateBlockProfile["record"]]) + "\n\n"

    outputFile.write(model_running_string)
    outputFile.write("\n".join(generateBlockProfile["plot"]) + "\n\n")
    outputFile.write("# Finish Generation #\n")
    outputFile.close()


model_file = open("./generate.json", mode="r", encoding="utf-8")
model_json = json.load(fp=model_file)
generateModel(model_json, "./generate_test.py")