import os
import collections
from viz_abstraction.block import Block
from model_profiling.profiler import BlockProfiler
import torch
from viz_api.viz_pytorch_api import input as input_Torch
from viz_api.viz_pytorch_api import layer as layer_Torch
from viz_api.viz_pytorch_api import monitor as monitor_Torch
from viz_api.viz_pytorch_api import transform as transform_Torch
from viz_api.viz_pytorch_api import optimizer as optimizer_Torch
from viz_api.viz_pytorch_api import node as VizNode_Torch
# -------------------- define model monitor (mode, saving, device) -------------------- #
model_running_train = True

monitor_Manager_0_dict = {'epochs': 15, 'model_save_path': './', 'device_name': 'cuda:0', 'name': 'monitor_Manager_0'}
monitor_Manager_0 = monitor_Torch.MonitorFinal_Torch(**monitor_Manager_0_dict)

# -------------------- load pretrained or system model -------------------- #
loadPath = "../system"
globalImportDict = dict()

def loadLayerNode(name, source):
	if os.path.exists(os.path.join(loadPath, name)):
		globalImportDict[name] = torch.load(os.path.join(loadPath, name))
	elif source != "" and os.path.exists(os.path.join(loadPath, source)):
		globalImportDict[name] = torch.load(os.path.join(loadPath, source))
	else:
		globalImportDict[name] = None

loadLayerNode('layer_Linear_0', '')
loadLayerNode('layer_ReLU_0', '')
loadLayerNode('layer_Linear_1', '')
loadLayerNode('layer_ReLU_1', '')
loadLayerNode('layer_Linear_2', '')
loadLayerNode('layer_LogSoftMax_0', '')
loadLayerNode('layer_NLLLoss_0', '')



# -------------------- define model node structure -------------------- #
layer_Linear_0_dict = {'in_features': 784, 'out_features': 128, 'inplace_forward': True, 'evaluate': False, 'bias': True, 'name': 'layer_Linear_0', 'device': monitor_Manager_0.device, 'import_layer': globalImportDict['layer_Linear_0']}
layer_Linear_0 = VizNode_Torch.LayerNode_Torch(layer_Torch.Linear_Torch, layer_Linear_0_dict)

layer_ReLU_0_dict = {'evaluate': False, 'name': 'layer_ReLU_0', 'device': monitor_Manager_0.device, 'import_layer': globalImportDict['layer_ReLU_0']}
layer_ReLU_0 = VizNode_Torch.LayerNode_Torch(layer_Torch.ReLU_Torch, layer_ReLU_0_dict)

layer_Linear_1_dict = {'in_features': 128, 'out_features': 64, 'inplace_forward': True, 'evaluate': False, 'bias': True, 'name': 'layer_Linear_1', 'device': monitor_Manager_0.device, 'import_layer': globalImportDict['layer_Linear_1']}
layer_Linear_1 = VizNode_Torch.LayerNode_Torch(layer_Torch.Linear_Torch, layer_Linear_1_dict)

layer_ReLU_1_dict = {'evaluate': False, 'name': 'layer_ReLU_1', 'device': monitor_Manager_0.device, 'import_layer': globalImportDict['layer_ReLU_1']}
layer_ReLU_1 = VizNode_Torch.LayerNode_Torch(layer_Torch.ReLU_Torch, layer_ReLU_1_dict)

layer_Linear_2_dict = {'in_features': 64, 'out_features': 10, 'inplace_forward': True, 'evaluate': False, 'bias': True, 'name': 'layer_Linear_2', 'device': monitor_Manager_0.device, 'import_layer': globalImportDict['layer_Linear_2']}
layer_Linear_2 = VizNode_Torch.LayerNode_Torch(layer_Torch.Linear_Torch, layer_Linear_2_dict)

layer_LogSoftMax_0_dict = {'dim': 1, 'inplace_forward': True, 'evaluate': False, 'name': 'layer_LogSoftMax_0', 'device': monitor_Manager_0.device, 'import_layer': globalImportDict['layer_LogSoftMax_0']}
layer_LogSoftMax_0 = VizNode_Torch.LayerNode_Torch(layer_Torch.LogSoftmax_Torch, layer_LogSoftMax_0_dict)

layer_NLLLoss_0_dict = {'evaluate': False, 'reduction': 'mean', 'name': 'layer_NLLLoss_0', 'device': monitor_Manager_0.device, 'import_layer': globalImportDict['layer_NLLLoss_0']}
layer_NLLLoss_0 = VizNode_Torch.LayerNode_Torch(layer_Torch.NLLLoss_Torch, layer_NLLLoss_0_dict)

input_MNIST_0_dict = {'root': '../dataset', 'max_batch_size': 64, 'shuffle': True, 'train': True, 'download': True, 'name': 'input_MNIST_0', 'device': monitor_Manager_0.device}
input_MNIST_0 = VizNode_Torch.MnistNode_Torch(input_Torch.MnistDataSetLoader_Torch, input_MNIST_0_dict)

transform_Flatten_0_dict = {'inplace_forward': False, 'name': 'transform_Flatten_0'}
transform_Flatten_0 = VizNode_Torch.TransformNode_Torch(transform_Torch.FlatTransform_Torch, transform_Flatten_0_dict)

if not model_running_train:
	layer_Linear_0.set_as_eval()
	layer_ReLU_0.set_as_eval()
	layer_Linear_1.set_as_eval()
	layer_ReLU_1.set_as_eval()
	layer_Linear_2.set_as_eval()
	layer_LogSoftMax_0.set_as_eval()
	layer_NLLLoss_0.set_as_eval()


# -------------------- model forwarding initialize -------------------- #
layer_Linear_0.set_output_port(1)
layer_ReLU_0.set_output_port(1)
layer_Linear_1.set_output_port(1)
layer_ReLU_1.set_output_port(1)
layer_Linear_2.set_output_port(1)
layer_LogSoftMax_0.set_output_port(1)
layer_NLLLoss_0.set_output_port(1)
input_MNIST_0.set_output_port(1)
transform_Flatten_0.set_output_port(1)

# -------------------- optimize initial -------------------- #
optimizer_SGD_0_track_list = [{'object':layer_Linear_2.get_linked_layer()}, {'object':layer_Linear_1.get_linked_layer()}, {'object':layer_Linear_0.get_linked_layer()}, ]
optimizer_SGD_0_dict = {'name': 'optimizer_SGD_0', 'learning_rate': 0.03, 'momentum': 0.9, 'object_to_track_list': optimizer_SGD_0_track_list}
optimizer_SGD_0 = optimizer_Torch.SGD_Torch(**optimizer_SGD_0_dict)

# -------------------- model block initialize -------------------- #
block_profiling_dict = collections.defaultdict(lambda: list())
Block0_nodes = [layer_Linear_0, layer_Linear_1, layer_Linear_2, ]
Block0 = Block(Block0_nodes, 'Block0')

for model_running_epoch in range(monitor_Manager_0.epochs):
	for iteration in range(input_MNIST_0.get_linked_input().get_number_batch()):
		# -------------------- model optimize -------------------- #
		if model_running_train:
			optimizer_SGD_0.clear_gradient()

		# -------------------- block profile -------------------- #
		Block0.update_record()

		# -------------------- model running -------------------- #
		input_MNIST_0.forward([iteration])

		input_MNIST_0_5_layer_NLLLoss_0_2 = input_MNIST_0.get_output_label_single(0)
		input_MNIST_0_4_transform_Flatten_0_1 = input_MNIST_0.get_output_tensor_single(0)
		transform_Flatten_0.forward([input_MNIST_0_4_transform_Flatten_0_1])

		transform_Flatten_0_4_layer_Linear_0_1 = transform_Flatten_0.get_output_tensor_single(0)
		layer_Linear_0.forward([transform_Flatten_0_4_layer_Linear_0_1])

		layer_Linear_0_4_layer_ReLU_0_1 = layer_Linear_0.get_output_tensor_single(0)
		layer_ReLU_0.forward([layer_Linear_0_4_layer_ReLU_0_1])

		layer_ReLU_0_4_layer_Linear_1_1 = layer_ReLU_0.get_output_tensor_single(0)
		layer_Linear_1.forward([layer_ReLU_0_4_layer_Linear_1_1])

		layer_Linear_1_4_layer_ReLU_1_1 = layer_Linear_1.get_output_tensor_single(0)
		layer_ReLU_1.forward([layer_Linear_1_4_layer_ReLU_1_1])

		layer_ReLU_1_4_layer_Linear_2_1 = layer_ReLU_1.get_output_tensor_single(0)
		layer_Linear_2.forward([layer_ReLU_1_4_layer_Linear_2_1])

		layer_Linear_2_4_layer_LogSoftMax_0_1 = layer_Linear_2.get_output_tensor_single(0)
		layer_LogSoftMax_0.forward([layer_Linear_2_4_layer_LogSoftMax_0_1])

		layer_LogSoftMax_0_4_layer_NLLLoss_0_1 = layer_LogSoftMax_0.get_output_tensor_single(0)
		layer_NLLLoss_0.forward([layer_LogSoftMax_0_4_layer_NLLLoss_0_1, input_MNIST_0_5_layer_NLLLoss_0_2])


		# -------------------- model optimize -------------------- #
		if model_running_train:
			layer_NLLLoss_0_4_optimizer_SGD_0_2 = layer_NLLLoss_0.get_output_tensor_single(0)
			optimizer_SGD_0.link_loss_tensor(layer_NLLLoss_0_4_optimizer_SGD_0_2)
			optimizer_SGD_0.backward()
			optimizer_SGD_0.step()

		# -------------------- block profile -------------------- #
		Block0.update_record()

	# -------------------- block epoch record -------------------- #
	block_profiling_dict['Block0'].append(Block0.get_meta_record())
	Block0.clear_meta_for_next_epoch()

# -------------------- block epoch plot -------------------- #
model_block_profiler = BlockProfiler('./')
model_block_profiler.generateBlockImage(block_profiling_dict)

# Finish Generation #
