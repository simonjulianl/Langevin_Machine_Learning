import torch
import torch.nn as nn

class optical_flow_concat2fields(nn.Module):
    def __init__(self, modelA, modelB):
        super(optical_flow_concat2fields, self).__init__()
        self.modelA = modelA
        self.modelB = modelB

    def forward(self, x):
        x1 = self.modelA(x)
        x2 = self.modelB(x1)
        return x2