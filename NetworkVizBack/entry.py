import torch
import torch.nn as nn

from test.model import Model


class ModelWithLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Model()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input, target):
        output = self.model(input)
        return self.loss_fn(output, target)


def skyline_model_provider():
    # Return a GPU-based instance of our model (that returns a loss)
    return ModelWithLoss().cuda()


def skyline_input_provider(batch_size=32):
    # Return GPU-based inputs for our model
    return (
      torch.randn((batch_size, 3, 256, 256)).cuda(),
      torch.randint(low=0, high=9, size=(batch_size,)).cuda(),
    )


def skyline_iteration_provider(model):
    # Return a function that executes one training iteration
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    def iteration(*inputs):
        optimizer.zero_grad()
        out = model(*inputs)
        out.backward()
        optimizer.step()
    return iteration