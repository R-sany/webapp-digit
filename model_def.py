import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(CNNGenerator, self).__init__()
        self.fc1 = nn.Linear(10, 256)
        self.deconv1 = nn.ConvTranspose2d(256, 128, 7, 1, 0)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.deconv3 = nn.ConvTranspose2d(64, 1, 4, 2, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = x.view(-1, 256, 1, 1) # Reshape for deconvolution
        x = self.deconv1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.relu(x)
        x = self.deconv3(x)
        x = self.tanh(x)
        return x
