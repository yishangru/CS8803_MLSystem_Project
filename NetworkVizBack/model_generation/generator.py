

class GlobalManager(object):
    def __init__(self, name: str = "GlobalManager"):
        super(GlobalManager, self).__init__()
        self.name = name
        self.block_id = 0
        self.node_id = 0

    def get_block_id(self):
        block_id_return = self.block_id
        self.block_id += 1
        return block_id_return

    def get_node_id(self):
        node_id_return = self.node_id
        self.node_id += 1
        return node_id_return

"""
1. forward (model structure, epoch + mode)
2. load dataset (load data) - input
3. load model (address)
constraint checker

[
    {
        name: --,
        type: --,
        description: --,
        parameters: [
            {
                ParaName: --,
                ParaClass: --,
                ParaValue: --,
                ParaRequired: --
            }
        ]
    }
]
#
[
    {
        id: number,
        type: 
        parameters: [
            {
                ParaName:
                ParaClass:
                ParaValue:
                ParaRequired:
            }
        ]
        links: [
            {
                
                
            }
        ]     
    }
]
"""

import inspect
nodeParameterList = list()
excludeDict = {"self", "device", "import_layer"}
node = {"name":"LogSoftmax", "type": "Layer", "api": "torch", "description": "description", "parameters": list()}
signature = inspect.signature(LogSoftmax_Torch.__init__)
for param in signature.parameters.values():
    if param.name not in excludeDict:
        paraDict = {"ParaName": param.name, "ParaClass": param.annotation.__name__}
        if param.default is param.empty:
            paraDict["ParaValue"] = ""
            paraDict["Required"] = 1
        else:
            paraDict["ParaValue"] = param.default
            paraDict["Required"] = 0
        node["Parameters"].append(paraDict)

# link relation


# parameter list
device = torch.device("cuda:0")
import_layer = None
GenerateDict = dict()
for param in node["Parameters"]:
    if param["ParaName"] == "dim":
        param["ParaValue"] = 1
    GenerateDict[param["ParaName"]] = param["ParaValue"]
GenerateDict["device"] = device
GenerateDict["import_layer"] = None
print(GenerateDict)
m = LogSoftmax_Torch(**GenerateDict)
input = Tensor_Torch(torch.randn(2, 3, device=device))
output = m.forward(input)

print(input.get_device(), m.get_device(), output.get_device())
print(input.name, "---", input.get_self_memory_size(), "---", input.get_grad_memory_size())
print(m.name, "---", m.get_feature_memory_size(), "---", m.get_grad_memory_size())
print(output.name, "---", output.get_self_memory_size(), "---", output.get_grad_memory_size())
print(torch.eq(input.get_linked_tensor(), output.get_linked_tensor()))