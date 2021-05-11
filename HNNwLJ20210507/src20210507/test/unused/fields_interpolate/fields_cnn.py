from HNN.models.fields_unet_parts import *
import torch.nn as nn
import torch

class fields_cnn(nn.Module):

    def __init__(self, gridL , batch_size, cnn_channels): # HK,
        super(fields_cnn, self).__init__()

        self.gridL = gridL

        self.conv1 = Conv(gridL, batch_size, 2, cnn_channels)
        self.conv2 = Conv(gridL, batch_size, cnn_channels, 2*cnn_channels)
        self.down1 = Down(gridL, batch_size,  2*cnn_channels,  4*cnn_channels)
        self.down2 = Down(gridL, batch_size,  4*cnn_channels,  8*cnn_channels)
        self.down3 = Down(gridL, batch_size,  8*cnn_channels, 16*cnn_channels)
        self.down4 = Down(gridL, batch_size, 16*cnn_channels, 32*cnn_channels)

        self.fc    = FC(32*cnn_channels, 32*cnn_channels)

        self.up1   = Up(gridL, batch_size, 2*32*cnn_channels, 16*cnn_channels, 4)
        self.up2   = Up(gridL, batch_size, 2*16*cnn_channels,  8*cnn_channels, 2)
        self.up3   = Up(gridL, batch_size,  2*8*cnn_channels,  4*cnn_channels, 2)
        self.up4   = Up(gridL, batch_size,  2*4*cnn_channels,  2*cnn_channels, 2)
        self.up5   = Up(gridL, batch_size,  2*2*cnn_channels,  2*cnn_channels, 2)
        self.conv3 = Conv(gridL, batch_size, 2*cnn_channels, cnn_channels)
        self.outc  = Conv(gridL, batch_size, cnn_channels, 2)

        # self.correction_term.apply(self.init_weights)
        print('fields_cnn initialized : ... ', 'gridL', self.gridL)
    # # ============================================
    # def init_weights(self,layer):
    #     if type(layer) == nn.Linear:
    #         nn.init.xavier_normal_(layer.weight,gain=0.1) # SJ make weights small
    #         layer.bias.data.fill_(0.0)
    # ============================================
    def get_gridL(self):
        return self.gridL
    # ============================================
    def forward(self, x, tau):

        batch_size, nchannels, gridx, gridy = x.shape
        # x.shape = [nsamples, nchannels=2, gridL, gridL]

        # down
        # print('down')
        x1 = self.conv1(x)
        # print(x1.shape)
        x2 = self.conv2(x1)
        # print(x2.shape)
        x3 = self.down1(x2)
        # print(x3.shape)
        x4 = self.down2(x3)
        # print(x4.shape)
        x5 = self.down3(x4)
        # print(x5.shape)
        x6 = self.down4(x5)
        # print(x6.shape)

        # bottom layer of unet
        x = self.fc(x6, tau, batch_size)
        # print('bottom layer')
        # print(x.shape)
        # print('up')
        # up
        x = self.up1(x, x6)
        # print(x.shape)
        x = self.up2(x, x5)
        # print(x.shape)
        x = self.up3(x, x4)
        # print(x.shape)
        x = self.up4(x, x3)
        # print(x.shape)
        x = self.up5(x, x2)
        # print(x.shape)
        x = self.conv3(x)
        # print(x.shape)
        out = self.outc(x)
        # out.shape = [batch_size, DIM=(fx,fy), gridLx, gridLy]

        return out
