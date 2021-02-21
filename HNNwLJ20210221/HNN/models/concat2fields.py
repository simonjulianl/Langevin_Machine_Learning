import torch
import torch.nn as nn

class concat2fields(nn.Module):
    def __init__(self, modelA, modelB):
        super(concat2fields, self).__init__()
        self.modelA = modelA
        self.modelB = modelB

    def forward(self, x1, x2):
        x1 = self.modelA(x1)
        x2 = self.modelB(x2)
        x = torch.cat((x1,x2),dim=1)
        print('phi field',x1.shape)
        print('p field',x2.shape)
        print('concat',x.shape)
        return x