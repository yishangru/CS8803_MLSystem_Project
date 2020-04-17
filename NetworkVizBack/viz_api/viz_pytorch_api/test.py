import torch
import viz_api.viz_pytorch_api.input as input
import viz_api.viz_pytorch_api.layer as layer
import viz_api.viz_pytorch_api.tensor as tensor
import viz_api.viz_pytorch_api.optimizer as optimizer
import viz_api.viz_pytorch_api.transform as transform

# device
device = torch.device("cuda:0")

# model layer structure
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

# model transform structure
flattransform_1 = transform.FlatTransform_Torch(inplace_forward=True)

# model input
root="../../static/dataset/"
shuffle=True
train=True
download=True
device=device
model_input = input.MnistDataSetLoader_Torch(root="../../static/dataset/", max_batch_size=64, shuffle=True, train=True, download=True, device=device)

# model optimizer
model_optimizer_1_track_list = [{"object": linear_1}, {"object": relu_1}, {"object": linear_2}, {"object": relu_2}, {"object": linear_3}, {"object": logsoftmax_1}]
model_optimizer_1 = optimizer.SGD_Torch(object_to_track_list=model_optimizer_1_track_list, learning_rate=0.003, momentum=0.9)

# monitor
epochs = 15
# ----- load model parameter ----- #
# ----- set the model iteration model (train or eval) ---- #

# model iteration
for epoch in range(epochs):
    print(epoch)
    running_loss = 0
    for iteration in range(model_input.get_number_batch()):
        model_optimizer_1.clear_gradient()

        # --------- training --------- topology generation 1 #
        data_iteration = model_input.get_loaded_tensor_img_single(iteration)
        flattransform_1_output = flattransform_1.forward(data_iteration)
        linear_1_output = linear_1.forward(flattransform_1_output)
        relu_1_output = relu_1.forward(linear_1_output)
        linear_2_output = linear_2.forward(relu_1_output)
        relu_2_output = relu_2.forward(linear_2_output)
        linear_3_output = linear_3.forward(relu_2_output)
        logsoftmax_1_output = logsoftmax_1.forward(linear_3_output)
        # --------- training --------- topology generation 1 #

        # --------- optimizer back generation --------- 1 #
        label = model_input.get_loaded_tensor_label_single(iteration)
        nllloss_1_output = nllloss_1.forward(logsoftmax_1_output, label)
        model_optimizer_1.link_loss_tensor(nllloss_1_output)
        model_optimizer_1.backward()
        model_optimizer_1.step()
        # --------- optimizer back generation --------- 1 #

        running_loss += nllloss_1_output.get_linked_tensor().item()
    print(running_loss)

# after training process
