# PyTorch

Range:
1. torch (torch.nn, torch.nn.functional, torch.Tensor, torch.autograd, torch.cuda, torch.optim)
2. torchvision.datasets
3. torchvision.transforms

Abstraction (model, block, layer, ...):
1. model (block)
2. block (layer)
3. layer (input, output, meta info)

Demo:
1. Mnist Hard-writing Task (change to block behavior) - basic functions, block behavior, hardware assignment, save and reload
2. Style Transfer Task (layer extraction for present models) - complex task, lower layer abstraction (extract the meta data from current models)

A Python-based scientific computing package
1. A replacement for Numpy to use the power of GPUs
2. A deep learning research platform that provides maximum flexibility and speed

## Computation Model (AutoGrad)
Graphical auto gradient operation (dataflow) - The *autograd* package provides automatic differentiation for all operations on Tensors -> backprop is defined by how your code is run. If the **.requirs_grad** is true, PyTorch will track all operations on it. When you finish your computation and call **.backward()** and have all the gradients computed automatically. The gradient for this tensor will be accumulated into **.grad** attribute. To stop a tensor from tracking history, you can call **.detach()** to detach it from the computation history, and to prevent future computation from being tracked. **.detach()** will create a new tensor with the same content but that does not require gradients. To prevent tracking history and using memory, you can wrap the code block in **with torch.no_grad()**: it can be helpful when evaluating a model because the model may have trainable parameters with **requires_grad=True**, while we don't need the gradients.

*Tensor* and *Function* are interconnected and build up an acyclic graph, that encodes a complete history of computation. Each tensor has a **.grad_fn** attribute that references a *Function* that has created the *Tensor* (except for Tensor created by user). We can create a tensor and set **requires_grad=True** to track computation with it.

Once calling the **.backward()** on the final tensor, corresponding related tensor will get the accumulated gradient.

## Neural Networks
Construct neural networks with **torch.nn** - nn depends on autograd to define models and differentiate them. An **nn.module** contains layers and a method **forward(input)** that returns the *output*. The typical training procedure for a neural network is as follows:
1. Define the neural network that has some learnable parameters (or weights) - layer (maxpool and relu not include)
2. Iterate over a dataset of inputs
3. Process the input through the network
4. Compute the loss (how far the output from being correct)
5. Propagate gradients back into the network's parameters (weights)
6. Update the weights of the network, typically using a simple update rule (weight = weight - learning_rate * gradient)

The first dimension of input will be the batch size. Loss as a scala function based on the network input (comparing with a fixed constant vector - classification)

## Input Type in Torch (value + shape + dtype)
1. Constant (torch.zeros(dim0, dim1, ..., dtype))
2. Random (torch.rand(dim0, dim1, ...))
3. Empty (torch.empty(dim0, dim1, ...))
4. Data input - torchvision (torchvision.datasets, torch.utils.data.DataLoader)

## Output in Torch
1. Model saver
2. Loss measure for network output

## Optimizer in Torch
1. Different update rules for network parameters
2. Clear the backward gradient (possible batching) - optimizer.zero_grad()

## Copy Tensor (value + shape + dtype -> new tensor)
1. Reshape (warning) - torch.view([shape array])
2. Change data type (warning)
3. Replace value (warning): value input

## Support Tensor Operation
1. Arithmetic operations (not inplace) - torch.add(x, y)
2. Arithmetic operations (inplace) - x.add_(y), xcopy_(y), x.t_(), x.add_(1)
3. trochvision.transforms

## Support Tensor Meta Information
1. torch.Size(torch)

## Support Data Type in Torch
1. torch.uint8 (for image)
2. torch.int8
3. torch.int16 (torch.short)
4. torch.int32 (torch.int)
5. torch.int64 (torch.long)
6. torch.float16 (torch.half)
7. torch.float32 (torch.float)
8. torch.float64 (torch.double)
9. torch.bool

## Tensor Place to GPU (Unified Call)
1. x.to(device=device, dtype=dtype)

# Future Project Extension
1. Add interface to support more present mainstream model
2. Support more self-defined code blocks for treatments
3. Add possible exception handling mechanism and more warning