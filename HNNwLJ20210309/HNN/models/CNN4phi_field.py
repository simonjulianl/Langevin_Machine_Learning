import torch
import torch.nn as nn
from .conv2d_PBC import compute_PBC_constants, compute_PBC
from MD_parameters import MD_parameters

class CNN4phi_field(nn.Module):

    def __init__(self, in_channels, out_channels):

        super(CNN4phi_field, self).__init__()

        initial_size = MD_parameters.npixels
        batch_size = MD_parameters.nsamples_ML

        self.PBC_constant = compute_PBC_constants(initial_size = initial_size, batch_size= batch_size, initial_channels=2)

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1),
            nn.ReLU()
        )


    def forward(self, x):
        # print('cnn4phi x',x.shape)
        x = compute_PBC(x, self.PBC_constant)
        # print('cnn4phi pbc x',x.shape)
        x = self.layer1(x)
        # print('cnn4phi x',x.shape)
        x = compute_PBC(x, self.PBC_constant)
        # print('cnn4phi pbc x',x.shape)
        out = self.layer2(x)
        # print('cnn4phi out',out.shape)
        return out