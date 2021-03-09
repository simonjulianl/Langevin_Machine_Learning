""" Full assembly of the parts to form the complete network """

from ML_parameters import ML_parameters
from .fields_unet_parts import *
from .concat2fields import concat2fields
from .CNN4p_field import CNN4p_field
from .CNN4phi_field import CNN4phi_field

class fields_unet(nn.Module):

    def __init__(self, bilinear=True):
        super(fields_unet, self).__init__()

        cnn_input = ML_parameters.cnn_input
        cnn_nhidden = ML_parameters.cnn_nhidden

        self.cat_channels = cnn_nhidden + cnn_nhidden
        self.bilinear = bilinear

        self.modelA = CNN4phi_field(cnn_input, cnn_nhidden)
        self.modelB = CNN4p_field(cnn_input, cnn_nhidden)

        self.conc = concat2fields(self.modelA, self.modelB)
        self.inc = DoubleConv(self.cat_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 128)
        self.fc = FC(128*4*4, 128*4*4)
        self.up1 = Up(256, 128 , bilinear)
        self.up2 = Up(256, 64, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.outc = OutConv(64, 2)

    def forward(self, phi_field, p_field , tau):

        x0 = self.conc(phi_field, p_field)
        x1 = self.inc(x0)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.fc(x3, tau)
        x = self.up1(x, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        print('logits', logits.shape)
        return logits