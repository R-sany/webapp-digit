import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # Example architecture — replace with your model!
        self.net = nn.Sequential(
            nn.Linear(10, 128),
            nn.ReLU(),
            nn.Linear(128, 784),  # assuming 28x28 images flattened
            nn.Tanh()
        )

    def forward(self, x):
        x = self.net(x)
        x = x.view(-1, 1, 28, 28)  # reshape to image format
        return x
