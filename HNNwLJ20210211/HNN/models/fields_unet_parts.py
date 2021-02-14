""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv2d_PBC import compute_PBC_constants, compute_PBC

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.PBC_constant = compute_PBC_constants(initial_size=32, batch_size=1, initial_channels=2)

        self.double_conv_first = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3),
            nn.ReLU(inplace=True))

        self.double_conv_second = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        x = compute_PBC(x, self.PBC_constant)
        x = self.double_conv_first(x)
        x = compute_PBC(x, self.PBC_constant)
        out = self.double_conv_second(x)

        return out

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class FC(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2))

        self.fully_connected = nn.Sequential(
            nn.Linear(in_channels+1, out_channels),
            nn.Tanh(),
            nn.Linear(out_channels, out_channels),
            nn.Tanh()
        )


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x, tau):

        x1 = self.maxpool_conv(x)
        print('maxpool',x1.shape)
        x2 = x1.view(x1.size(0), self.num_flat_features(x1))
        print('flatten',x2.shape)
        x2 = torch.cat([x2, tau], dim=1)
        print('concat',x2.shape)
        x2 = self.fully_connected(x2)
        x3 = x2.view(x2.size(0),x1.shape[1],x1.shape[2],x1.shape[3])

        return x3

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        print('up', x1.shape)
        x1 = self.up(x1)
        print('up', x1.shape)

        # input is CHW
        # diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        # diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
        #
        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # # if you have padding issues, see
        # # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)