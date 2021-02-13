""" Full assembly of the parts to form the complete network """

from .fields_unet_parts import *
from .concat2fields import concat2fields
from .CNN4p_field import CNN4p_field
from .CNN4phi_field import CNN4phi_field

class fields_unet(nn.Module):

    def __init__(self, n_channels, n_classes, bilinear=True):
        super(fields_unet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.modelA = CNN4phi_field()
        self.modelB = CNN4p_field()

        self.conc = concat2fields(self.modelA, self.modelB)
        self.inc = DoubleConv(64, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 128)
        self.up1 = Up(256, 128, bilinear)
        self.up2 = Up(192, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, phi_field, p_field):

        x0 = self.conc(phi_field, p_field)
        x1 = self.inc(x0)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        logits = self.outc(x)
        print('logits', logits.shape)
        return logits