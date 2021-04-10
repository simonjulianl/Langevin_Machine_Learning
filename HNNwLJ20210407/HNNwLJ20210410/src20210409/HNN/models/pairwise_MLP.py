import torch.nn as nn
import torch

class pairwise_MLP(nn.Module):

    def __init__(self, MLP_input, MLP_nhidden):
        super(pairwise_MLP, self).__init__()

        self.correction_term = nn.Sequential(
            nn.Linear(MLP_input, MLP_nhidden),
            nn.Tanh(),
            nn.Linear(MLP_nhidden, MLP_nhidden),
            nn.Tanh(),
            nn.Linear(MLP_nhidden, 2)
        )
        self.correction_term.double()
    # ============================================
    def init_weights(self,layer):
        if type(layer) == nn.Linear:
            nn.init.xavier_normal_(layer.weight)
            layer.bias.data.fill_(0.01)
    # ============================================
    def forward(self, x):

        self.correction_term.apply(self.init_weights)
        MLdHdq = self.correction_term(x)
        # print('w 1',self.correction_term[0].weight)
        # print('w 2', self.correction_term[2].weight)
        print('w 3', self.correction_term[4].weight)
        # print('ML output',MLdHdq)

        return MLdHdq
