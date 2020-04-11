import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3)
        self.linear = nn.Linear(in_features=387096, out_features=10)

    def forward(self, input):
        out = self.conv(input)
        return self.linear(out.view(-1, 387096))