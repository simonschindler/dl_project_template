import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.layer = nn.Linear(128, num_classes)

    def forward(self, x):
        return self.layer(x)
