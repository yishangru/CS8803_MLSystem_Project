# Project Design

## Abstraction
### block level
A block is viewed as a combination of multiple layers. The input and output relations between multiple layer, along with the possible transform are defined in this level. A block doesn't link with optimizer or loss function, while we can perform transform on the output to act as loss function.

### model level
A model is viewed as a combination of multiple blocks. The input and output relations between multiple blocks are defined in this level, with the loss function and optimizer update. The iteration method is also defined in this level.

## Input Output Transform

## Code Structure
### abstraction
This directory is for the block and model level abstraction.

### platform
This directory is the middleware for transform the platform api (PyTorch) to our code.