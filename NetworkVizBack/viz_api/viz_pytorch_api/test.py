import os
import viz_api.viz_pytorch_api.input as input
import viz_api.viz_pytorch_api.layer as layer
import viz_api.viz_pytorch_api.tensor as tensor
import viz_api.viz_pytorch_api.optimizer as optimizer
import viz_api.viz_pytorch_api.transform as transform
import viz_api.viz_pytorch_api.monitor as monitor

######### ----------------------- USER SPACE ----------------------- #########
"""
Define the meta information in monitor:
    epochs: number of dataset epoch (int)
    device_name: device to place parameter and tensor (str), i.e. "cpu", "cuda:0"
    model_save_path: directory for save model (str)
"""

# model monitor #
MonitorFinal = monitor.MonitorFinal_Torch(epochs=10, model_save_path="../../static/saved_model", device_name="cuda:0")
if os.path.exists(MonitorFinal.model_save_path):
    if not os.path.isdir(MonitorFinal.model_save_path):
        print("Please change the model_save_path to a directory path")
        raise RuntimeError
else:
    os.mkdir(MonitorFinal.model_save_path)
device = MonitorFinal.device
# model monitor #

"""
Define the dataset for model iteration - can be training or evaluation dataset (should have same structure)
"""
# model input #
model_input = input.MnistDataSetLoader_Torch(root="../../static/dataset/", max_batch_size=64, shuffle=True, train=True, download=True, device=device)
# model input #

"""
Define the path to load pre-trained or finished trained model. The initial path is set as the default path to load for 
pre-trained model. User can use the new generated model (change following path to the one that your model is saved).
"""
# load previous model or saved model (VGG) #

# load previous model or saved model (VGG) #

######### ----------------------- USER SPACE ----------------------- #########



######### ----------------------- GENERATION SPACE ----------------------- #########
# model layer structure #
input_size = 784
output_size = 10
hidden_sizes = [128, 64]
linear_1 = layer.Linear_Torch(in_features=input_size, out_features=hidden_sizes[0], inplace_forward=False, device=device)
relu_1 = layer.ReLU_Torch(device=device)
linear_2 = layer.Linear_Torch(in_features=hidden_sizes[0], out_features=hidden_sizes[1], inplace_forward=True, device=device)
relu_2 = layer.ReLU_Torch(device=device)
linear_3 = layer.Linear_Torch(in_features=hidden_sizes[1], out_features=output_size, inplace_forward=True, device=device)
logsoftmax_1 = layer.LogSoftmax_Torch(dim=1, inplace_forward=True, device=device)
nllloss_1 = layer.NLLLoss_Torch(device=device)
# model layer structure #


# model transform structure #
flattransform_1 = transform.FlatTransform_Torch(inplace_forward=True)
# model transform structure #


# model optimizer #
model_optimizer_1_track_list = [{"object": linear_1}, {"object": relu_1}, {"object": linear_2}, {"object": relu_2}, {"object": linear_3}, {"object": logsoftmax_1}]
model_optimizer_1 = optimizer.SGD_Torch(object_to_track_list=model_optimizer_1_track_list, learning_rate=0.003, momentum=0.9)
# model optimizer #


# model training #

# model iteration #
for epoch in range(MonitorFinal.epochs):
    print(epoch)
    running_loss = 0
    for iteration in range(model_input.get_number_batch()):
        model_optimizer_1.clear_gradient()

        # --------- training --------- #
        data_iteration = model_input.get_loaded_tensor_img_single(iteration)
        flattransform_1_output = flattransform_1.forward(data_iteration)
        linear_1_output = linear_1.forward(flattransform_1_output)
        relu_1_output = relu_1.forward(linear_1_output)
        linear_2_output = linear_2.forward(relu_1_output)
        relu_2_output = relu_2.forward(linear_2_output)
        linear_3_output = linear_3.forward(relu_2_output)
        logsoftmax_1_output = logsoftmax_1.forward(linear_3_output)
        # --------- training --------- topology generation 1 #

        # --------- optimization --------- 1 #
        label = model_input.get_loaded_tensor_label_single(iteration)
        nllloss_1_output = nllloss_1.forward(logsoftmax_1_output, label)
        model_optimizer_1.link_loss_tensor(nllloss_1_output)
        model_optimizer_1.backward()
        model_optimizer_1.step()
        # --------- optimization --------- 1 #

        running_loss += nllloss_1_output.get_linked_tensor().item()
    print(running_loss)
# model iteration #

# after training process #


# after training process #


# model training #


# model evaluation #

# model evaluation #

######### ----------------------- GENERATION SPACE ----------------------- #########