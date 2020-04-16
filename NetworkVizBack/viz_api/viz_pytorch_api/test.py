import torch
import viz_api.viz_pytorch_api.input as input
import viz_api.viz_pytorch_api.layer as layer
import viz_api.viz_pytorch_api.tensor as tensor
import viz_api.viz_pytorch_api.optimizer as optimizer

epochs = 15
input_size = 784
output_size = 10
hidden_sizes = [128, 64]
device = torch.device("cuda:0")

# model layer structure
linear1 = layer.Linear_Torch(in_features=input_size, out_features=hidden_sizes[0], inplace_forward=False, device=device)
relu1 = layer.ReLU_Torch(device=device)
linear2 = layer.Linear_Torch(in_features=hidden_sizes[0], out_features=hidden_sizes[1], inplace_forward=True, device=device)
relu2 = layer.ReLU_Torch(device=device)
linear3 = layer.Linear_Torch(in_features=hidden_sizes[1], out_features=output_size, inplace_forward=True, device=device)
logsoftmax1 = layer.LogSoftmax_Torch(dim=1, inplace_forward=True, device=device)
nllloss1 = layer.NLLLoss_Torch(device=device)

# model input
root="../../static/dataset/"
shuffle=True
train=True
download=True
device=device
model_input = input.MnistDataSetLoader_Torch(root="../../static/dataset/", max_batch_size=64, shuffle=True, train=True, download=True, device=device)

# model optimizer
model_optimizer_1_track_list = [{"object": linear1}, {"object": relu1}, {"object": linear2}, {"object": relu2}, {"object": linear3}, {"object": logsoftmax1}]
model_optimizer_1 = optimizer.SGD_Torch(object_to_track_list=model_optimizer_1_track_list, learning_rate=0.003, momentum=0.9)

# model iteration
for epoch in range(epochs):
    print(epoch)
    running_loss = 0
    for iteration in range(model_input.get_number_batch()):
        model_optimizer_1.clear_gradient()

        # --------- training --------- topology generation 1 #
        image = model_input.get_loaded_tensor_img_single(iteration)
        image.set_linked_tensor(image.get_linked_tensor().view(image.get_linked_tensor().shape[0], -1))
        linear1_output = linear1.forward(image)
        relu1_output = relu1.forward(linear1_output)
        linear2_output = linear2.forward(relu1_output)
        relu2_output = relu2.forward(linear2_output)
        linear3_output = linear3.forward(relu2_output)
        logsoftmax1_output = logsoftmax1.forward(linear3_output)
        # --------- training --------- topology generation 1 #

        # --------- optimizer back generation --------- 1 #
        label = model_input.get_loaded_tensor_label_single(iteration)
        nllloss1_output = nllloss1.forward(logsoftmax1_output, label)
        model_optimizer_1.link_loss_tensor(nllloss1_output)
        model_optimizer_1.backward()
        model_optimizer_1.step()
        # --------- optimizer back generation --------- 1 #

        running_loss += nllloss1_output.get_linked_tensor().item()
    print(running_loss)