# Project Design
## System Architecture
1. Backend once starts, generate the protocol file for front end to get the parameter and generation type
2. Front end read the proptocol and generate node with input parameter
3. After the network structure is finished, select Platform and generate code
4. Iteration running with meta information profiling (start profiling)

## Training and Evaluation

## Abstraction
### node level
A node can be the wrapper for layer, input, transform. They all support for multiple output (each output is a separated copy). As for input wrapper, it doesn't take tensor as input.

### block level
A block is viewed as a combination of multiple nodes. The input and output relations between multiple nodes, along with the possible transform are defined in this level. A block doesn't link with optimizer or loss function, while we can perform transform on the output to act as loss function.
1. Processing block (not gradient update)
2. Staging block (we need gradient update for block weight)

### model level
A model is viewed as a combination of multiple blocks. The input and output relations between multiple blocks are defined in this level, with the loss function and optimizer update. The iteration method is also defined in this level.

## Input Output Transform

## Code Structure
### abstraction
This directory is for the block and model level abstraction.

### platform
This directory is the middleware for transform the platform api (PyTorch) to our code. Wrapper is the interface for the method abstraction.