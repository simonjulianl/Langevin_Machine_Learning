""" Parts of the U-Net model """

import torch
import torch.nn as nn
from HNN.models.conv2d_PBC import compute_PBC_constants, compute_PBC

class Conv(nn.Module):
    """ apply pbc padding and define a 2D convolution layer """

    def __init__(self, gridL, batch_size, in_channels, out_channels):
        '''
        Parameters
        ----------
        gridL : int
        batch_size : int
        '''

        super(Conv, self).__init__()

        self.PBC_constant = compute_PBC_constants(gridL, batch_size, in_channels)

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.LeakyReLU()) # HK

    def forward(self, x):

        x = compute_PBC(x, self.PBC_constant)
        # shape is [batch_size, channels, gridL+2, gridL+2] ; padding = 2

        out = self.conv_layer(x)
        # shape is [batch_size, channels, gridL, gridL]

        return out

class Down(nn.Module):
    """ Downscaling with maxpool then conv """

    def __init__(self, gridL, batch_size, in_channels, out_channels):

        super(Down, self).__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            Conv(gridL, batch_size, in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class FC(nn.Module):
    """ flatten with global avg pooling and feed it to fully connected layer,
    and then reshape for upsampling"""

    def __init__(self, in_channels, out_channels):

        super(FC, self).__init__()

        self.AdaptiveAvgPool2d = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)))

        self.fully_connected = nn.Sequential(
            nn.Linear(in_channels+1, out_channels), # +1 is tau
            nn.Tanh(),  # HK just linear better?
            nn.Linear(out_channels, out_channels),
            nn.Tanh()
        )

    def forward(self, x, tau, batch_size):

        x = self.AdaptiveAvgPool2d(x)
        # shape is [batch_size, channels, 1, 1]

        x1 = x.view(x.size(0),-1)
        # shape is [batch_size, channels]

        tau_tensor = torch.zeros([batch_size, 1])
        tau_tensor.fill_(tau)

        x2 = torch.cat([x1, tau_tensor], dim=1)
        # shape is [batch_size, channels+1]

        x2 = self.fully_connected(x2)
        x3 = x2.view(x2.size(0),x.shape[1],x.shape[2],x.shape[3])
        # shape is [batch_size, channels, 1, 1]

        return x3

class Up(nn.Module):
    """Upscaling then conv"""

    def __init__(self, gridL, batch_size, in_channels, out_channels, scale_factor):

        super(Up, self).__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.conv_up = Conv(gridL, batch_size, in_channels, out_channels)

    def forward(self, x1, x2):

        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        out = self.conv_up(x)

        return out
