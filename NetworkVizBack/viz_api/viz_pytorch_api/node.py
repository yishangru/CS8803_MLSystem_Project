from viz_api import node
from viz_api.viz_pytorch_api.tensor import Tensor_Torch


class LayerNode_Torch(node.LayerNode):
    def __init__(self, LayerClass, GenerateDict: dict):
        super(LayerNode_Torch, self).__init__(GenerateDict["name"])
        self.linkedLayer = LayerClass(**GenerateDict)
        self.linkedClass = type(self.linkedLayer).__name__
        self.outputMapping = list()

    # return linked layer
    def get_linked_layer(self):
        return self.linkedLayer

    # set as evaluation mode
    def set_as_eval(self):
        self.linkedLayer.set_as_eval()

    # set as train mode
    def set_as_training(self):
        self.linkedLayer.set_as_training()

    def set_output_port(self, number: int):
        for i in range(number):
            self.outputMapping.append(Tensor_Torch(linked_tensor=None, name=self.name + "_output_" + str(i+1)))

    # return the output mapping for tensor output
    def get_output_tensor_single(self, number: int):
        return self.outputMapping[number]

    def get_output_tensor_all(self):
        return self.outputMapping

    def forward(self, inputList: list):
        # prepare the data for underlying forward
        output_tensor = self.linkedLayer.forward(*inputList)
        self.outputMapping[0].set_linked_tensor(output_tensor)
        for i in range(1, len(self.outputMapping)):
            generated_tensor_copy = self.outputMapping[0].get_deep_copy()
            self.outputMapping[i].set_linked_tensor(generated_tensor_copy)

    # return KB in memory usage for the loaded tensor
    def get_tensor_memory_size(self):
        tensor_memory_usage = 0 if self.linkedLayer.inplace_forward else self.outputMapping[0].get_self_memory_size()
        for i in range(1, len(self.outputMapping)):
            tensor_memory_usage += self.outputMapping[i].get_self_memory_size()
        return tensor_memory_usage

    # return KB in memory usage for gradients of the loaded tensor
    def get_tensor_grad_memory_size(self):
        tensor_grad_usage = 0 if self.linkedLayer.inplace_forward else self.outputMapping[0].get_grad_memory_size()
        for i in range(1, len(self.outputMapping)):
            tensor_grad_usage += self.outputMapping[i].get_grad_memory_size()
        return tensor_grad_usage

    # return KB in memory usage for layer feature (weight, bias)
    def get_layer_feature_memory_size(self):
        return self.linkedLayer.get_feature_memory_size()

    # return KB in memory usage for layer gradients
    def get_layer_grad_memory_size(self):
        return self.linkedLayer.get_grad_memory_size()


class TransformNode_Torch(node.TransformNode):
    def __init__(self, TransformClass, GenerateDict: dict):
        super(TransformNode_Torch, self).__init__(GenerateDict["name"])
        self.linkedTransform = TransformClass(**GenerateDict)
        self.linkedClass = type(self.linkedTransform).__name__
        self.outputMapping = list()

    def set_output_port(self, number: int):
        for i in range(number):
            self.outputMapping.append(Tensor_Torch(linked_tensor=None, name=self.name + "_output_" + str(i+1)))

    # return the output mapping for tensor output
    def get_output_tensor_single(self, number: int):
        return self.outputMapping[number]

    def get_output_tensor_all(self):
        return self.outputMapping

    def forward(self, inputList: list):
        # prepare the data for underlying forward
        output_tensor = self.linkedTransform.forward(*inputList)
        self.outputMapping[0].set_linked_tensor(output_tensor)
        for i in range(1, len(self.outputMapping)):
            generated_tensor_copy = self.outputMapping[0].get_deep_copy()
            self.outputMapping[i].set_linked_tensor(generated_tensor_copy)

    # return KB in memory usage for the loaded tensor
    def get_tensor_memory_size(self):
        tensor_memory_usage = 0 if self.linkedTransform.inplace_forward else self.outputMapping[0].get_self_memory_size()
        for i in range(1, len(self.outputMapping)):
            tensor_memory_usage += self.outputMapping[i].get_self_memory_size()
        return tensor_memory_usage

    # return KB in memory usage for gradients of the loaded tensor
    def get_tensor_grad_memory_size(self):
        tensor_grad_usage = 0 if self.linkedTransform.inplace_forward else self.outputMapping[0].get_grad_memory_size()
        for i in range(1, len(self.outputMapping)):
            tensor_grad_usage += self.outputMapping[i].get_grad_memory_size()
        return tensor_grad_usage


class InputNode_Torch(node.InputNode):
    def __init__(self, InputClass, GenerateDict: dict):
        super(InputNode_Torch, self).__init__(GenerateDict["name"])
        self.linkedInput = InputClass(**GenerateDict)
        self.linkedClass = type(self.linkedInput).__name__
        self.outputMapping = list()

    # return linked input
    def get_linked_input(self):
        return self.linkedInput

    def set_output_port(self, number: int):
        for i in range(number):
            self.outputMapping.append(Tensor_Torch(linked_tensor=None, name=self.name + "_output_" + str(i + 1)))

    # return the output mapping for tensor output
    def get_output_tensor_single(self, number: int):
        return self.outputMapping[number]

    def get_output_tensor_all(self):
        return self.outputMapping

    def forward(self, inputList: list = None):
        # prepare the data for underlying forward
        output_tensor = self.linkedInput.get_loaded_tensor()
        self.outputMapping[0].set_linked_tensor(output_tensor)
        for i in range(1, len(self.outputMapping)):
            generated_tensor_copy = self.outputMapping[0].get_deep_copy()
            self.outputMapping[i].set_linked_tensor(generated_tensor_copy)

    # return KB in memory usage for the loaded tensor
    def get_tensor_memory_size(self):
        tensor_memory_usage = 0
        for i in range(1, len(self.outputMapping)):
            tensor_memory_usage += self.outputMapping[i].get_self_memory_size()
        tensor_memory_usage += self.linkedInput.get_tensor_memory_size()
        return tensor_memory_usage

    # return KB in memory usage for gradients of the loaded tensor
    def get_tensor_grad_memory_size(self):
        tensor_grad_usage = 0
        for i in range(1, len(self.outputMapping)):
            tensor_grad_usage += self.outputMapping[i].get_grad_memory_size()
        tensor_grad_usage += self.linkedInput.get_tensor_grad_memory_size()
        return tensor_grad_usage


# special treatment for Mnist Data (with label)
class MnistNode_Torch(node.InputNode):
    def __init__(self, InputClass, GenerateDict: dict):
        super(MnistNode_Torch, self).__init__(GenerateDict["name"])
        self.linkedInput = InputClass(**GenerateDict)
        self.linkedClass = type(self.linkedInput).__name__
        self.outputMapping = list()
        self.labelMapping = list()

    # return linked input
    def get_linked_input(self):
        return self.linkedInput

    def set_output_port(self, number: int):
        for i in range(number):
            self.outputMapping.append(Tensor_Torch(linked_tensor=None, name=self.name + "_output_" + str(i + 1)))
            self.labelMapping.append(Tensor_Torch(linked_tensor=None, name=self.name + "_label_" + str(i + 1)))

    # return the output mapping for tensor output
    def get_output_tensor_single(self, number: int):
        return self.outputMapping[number]

    def get_output_tensor_all(self):
        return self.outputMapping

    def forward(self, inputList: list):
        # prepare the data for underlying forward
        output_tensor = self.linkedInput.get_loaded_tensor_img_single(inputList[0])
        label_tensor = self.linkedInput.get_loaded_tensor_label_single(inputList[0])
        self.outputMapping[0].set_linked_tensor(output_tensor.get_linked_tensor())
        self.labelMapping[0].set_linked_tensor(label_tensor.get_linked_tensor())
        for i in range(1, len(self.outputMapping)):
            generated_tensor_copy = self.outputMapping[0].get_deep_copy()
            self.outputMapping[i].set_linked_tensor(generated_tensor_copy)
            generated_label_copy = self.labelMapping[0].get_deep_copy()
            self.labelMapping[i].set_linked_tensor(generated_label_copy)

    # return KB in memory usage for the loaded tensor
    def get_tensor_memory_size(self):
        tensor_memory_usage = 0
        for i in range(1, len(self.outputMapping)):
            tensor_memory_usage += self.outputMapping[i].get_self_memory_size()
            tensor_memory_usage += self.labelMapping[i].get_self_memory_size()
        tensor_memory_usage += self.linkedInput.get_tensor_memory_size()
        return tensor_memory_usage

    # return KB in memory usage for gradients of the loaded tensor
    def get_tensor_grad_memory_size(self):
        tensor_grad_usage = 0
        for i in range(1, len(self.outputMapping)):
            tensor_grad_usage += self.outputMapping[i].get_grad_memory_size()
            tensor_grad_usage += self.labelMapping[i].get_grad_memory_size()
        tensor_grad_usage += self.linkedInput.get_tensor_grad_memory_size()
        return tensor_grad_usage


class ConstantNode_Torch(node.ConstantNode):
    def __init__(self, ConstantClass, GenerateDict: dict):
        super(ConstantNode_Torch, self).__init__(GenerateDict["name"])
        self.linkedConstant = ConstantClass(**GenerateDict)
        self.linkedClass = type(self.linkedConstant).__name__
        self.outputMapping = list()

    # return linked constant
    def get_linked_constant(self):
        return self.linkedConstant

    def set_output_port(self, number: int):
        for i in range(number):
            self.outputMapping.append(Tensor_Torch(linked_tensor=None, name=self.name + "_output_" + str(i + 1)))

    # return the output mapping for tensor output
    def get_output_tensor_single(self, number: int):
        return self.outputMapping[number]

    def get_output_tensor_all(self):
        return self.outputMapping

    def forward(self, inputList: list):
        # prepare the data for underlying forward
        output_tensor = self.linkedConstant.get_saved_tensor()
        self.outputMapping[0].set_linked_tensor(output_tensor)
        for i in range(1, len(self.outputMapping)):
            generated_tensor_copy = self.outputMapping[0].get_deep_copy()
            self.outputMapping[i].set_linked_tensor(generated_tensor_copy)

    # return KB in memory usage for the loaded tensor
    def get_tensor_memory_size(self):
        tensor_memory_usage = 0
        for i in range(1, len(self.outputMapping)):
            tensor_memory_usage += self.outputMapping[i].get_self_memory_size()
        tensor_memory_usage += self.linkedConstant.get_tensor_memory_size()
        return tensor_memory_usage

    # return KB in memory usage for gradients of the loaded tensor
    def get_tensor_grad_memory_size(self):
        tensor_grad_usage = 0
        for i in range(1, len(self.outputMapping)):
            tensor_grad_usage += self.outputMapping[i].get_grad_memory_size()
        tensor_grad_usage += self.linkedConstant.get_tensor_grad_memory_size()
        return tensor_grad_usage


# --------------------- test node --------------------- #
def test_layer_node():
    import torch
    from viz_api.viz_pytorch_api import layer
    device = torch.device("cuda:0")

    GeneratedDict = {"in_features": 128,
                     "out_features": 10,
                     "inplace_forward": False,
                     "import_layer": None,
                     "name": "Linear",
                     "bias": True,
                     "device": device}
    linear1 = LayerNode_Torch(layer.Linear_Torch, GenerateDict=GeneratedDict)
    linear1.set_output_port(2)

    test_rand = Tensor_Torch(torch.randn(10, 128).to(device))
    linear1.forward([test_rand])
    for i in range(len(linear1.outputMapping)):
        outputTensor = linear1.outputMapping[i]
        print(outputTensor.name, id(outputTensor.get_linked_tensor()), outputTensor.get_linked_tensor().size())
    print(linear1.get_tensor_memory_size(), linear1.get_tensor_grad_memory_size())
    print(linear1.get_layer_feature_memory_size(), linear1.get_layer_grad_memory_size())


def test_transform_node():
    import torch
    from viz_api.viz_pytorch_api import transform
    device = torch.device("cuda:0")

    GeneratedDict = {"inplace_forward": True,
                     "name": "Add1"}
    one_tensor_input_1 = Tensor_Torch(torch.ones(1, 1).to(device))
    one_tensor_input_2 = Tensor_Torch(torch.ones(1, 1).to(device))
    one_tensor_input_3 = Tensor_Torch(torch.ones(1, 1).to(device))
    add1 = TransformNode_Torch(transform.AddTransform_Torch, GenerateDict=GeneratedDict)
    add1.set_output_port(2)

    add1.forward([one_tensor_input_1, one_tensor_input_2, one_tensor_input_3])
    print(one_tensor_input_1.get_linked_tensor())
    for i in range(len(add1.outputMapping)):
        outputTensor = add1.outputMapping[i]
        print(outputTensor.name, id(outputTensor.get_linked_tensor()), outputTensor.get_linked_tensor())

    GeneratedDict = {"inplace_forward": False,
                     "name": "Add1"}
    one_tensor_input_1 = Tensor_Torch(torch.ones(1, 1).to(device))
    one_tensor_input_2 = Tensor_Torch(torch.ones(1, 1).to(device))
    one_tensor_input_3 = Tensor_Torch(torch.ones(1, 1).to(device))
    add1 = TransformNode_Torch(transform.AddTransform_Torch, GenerateDict=GeneratedDict)
    add1.set_output_port(2)

    add1.forward([one_tensor_input_1, one_tensor_input_2, one_tensor_input_3])
    print(one_tensor_input_1.get_linked_tensor())
    for i in range(len(add1.outputMapping)):
        outputTensor = add1.outputMapping[i]
        print(outputTensor.name, id(outputTensor.get_linked_tensor()), outputTensor.get_linked_tensor())


#test_layer_node()
#test_transform_node()