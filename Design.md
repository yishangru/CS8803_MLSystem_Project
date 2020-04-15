# Project Design
## System Architecture
1. Backend once starts, generate the protocol file for front end to get the parameter and generation type
2. Front end read the proptocol and generate node with input parameter
3. After the network structure is finished, select Platform and generate code
4. Iteration running with meta information profiling (start profiling)

## Training and Evaluation


# Design Doc
## Module Define
### Tensor
Tensor is the data holder in our project. It is currently a wrapper for PyTroch tensor with additional information (e.g. name, operation) recorded. 

### Node
A node is the wrapper for layer, input, transform, with support for multiple output (the multiple outputs are the deep copy of the first output tensor).
1. **Transform** node: a transform node is constructed by a transform fuction and the corresponding output tensor given input tensor. It take tensor(s) as input and perform tensor transform as output. (support multiple inputs and multiple outputs).
2. **Layer** node: a layer node is constructed by a layer fuction and the corresponding output tensor given input tensor. It take tensor(s) as input and perform tensor operation as output. (loss layer support multiple inputs).
3. **Constant** node: a constant node is constructed by a input fuction for tensor loading and loaded tensor given input parameters (e.g. data path for tensor load, constant tensor). It take parameters as input and perform tensor generation and load.
3. **Input** node: a input node is a special constant node constructed by a input fuction for tensor loading and loaded tensor given input parameters (e.g. data path for tensor load, constant tensor). It take parameters as input and perform dataset load or tensor load as output. (dataset load not support mutiple outputs) - it is the starting point for our model structure generation (only one). As for dataset, we currently support MNIST (image, label). The input node for MNIST will have two output, one for image and one for corresponding label.

### Block
A block is the wrapper with similarity as **nn.module**. It is a combination of layer node (support multiple inputs and multiple outputs).

### Optimizer
Block and node can register in the optimizer for auto gradient update. For block registration, the parameters of inner layer will be added for gradient update. For node registration, the tensor in node will be added for gradient update. In order to perform gradient update, a loss function taking tensors for loss measure is required.

### Monitor
Monitor is for epoch check and possible model save. We currently support epoch based model saver.

### Profiling
Profiling is for the blocks and nodes. The overall profiling is for (input, transform) nodes and blocks (layer parameters).

## Abstraction
### block level
Depending on the usage, the parameters of block can be added to the optimizer for gradient updates.
1. Processing block (not gradient update)
2. Staging block (we need gradient update for block weight)

### model level
A model is viewed as a combination of multiple blocks and nodes. The input and output relations between multiple blocks and nodes are defined in this level (the starting node is the input node, we perform graph transverse to generate the model structure), with the loss function and optimizer update. The iteration method is also defined in this level. The generated model will have a forward method take data and mode as input. (mode is for training or eval) - If eval, the optimizer is suspended.


## Code Structure
### abstraction
This directory is for the block and model level abstraction.

### platform
This directory is the middleware for transform the platform api (PyTorch) to our code. Wrapper is the interface for the method abstraction.