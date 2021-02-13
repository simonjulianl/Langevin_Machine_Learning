import torch
import torch.nn as nn
from .conv2d_PBC import compute_PBC_constants, compute_PBC

class CNN4phi_field(nn.Module):

    def __init__(self):

        super(CNN4phi_field, self).__init__()

        self.PBC_constant = compute_PBC_constants(initial_size=32, batch_size=1, initial_channels=2)

        self.layer1 = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=1),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU()
        )


    def forward(self, x):
        print('cnn4phi x',x.shape)
        x = compute_PBC(x, self.PBC_constant)
        print('cnn4phi pbc x',x.shape)
        x = self.layer1(x)
        print('cnn4phi x',x.shape)
        x = compute_PBC(x, self.PBC_constant)
        print('cnn4phi pbc x',x.shape)
        out = self.layer2(x)
        print('cnn4phi out',out.shape)
        return out