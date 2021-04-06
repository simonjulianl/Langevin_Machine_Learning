import torch
import torch.nn as nn
from parameters.ML_parameters import ML_parameters

class pairwise_MLP(nn.Module):

    def __init__(self):
        super(pairwise_MLP, self).__init__()

        MLP_input = ML_parameters.MLP_input
        MLP_nhidden = ML_parameters.MLP_nhidden

        self.correction_term = nn.Sequential(
            nn.Linear(MLP_input, MLP_nhidden),
            nn.Tanh(),
            nn.Linear(MLP_nhidden, MLP_nhidden),
            nn.Tanh(),
            nn.Linear(MLP_nhidden, MLP_nhidden),
            nn.Tanh(),
            nn.Linear(MLP_nhidden, MLP_nhidden),
            nn.Tanh(),
            nn.Linear(MLP_nhidden, 2)
        )
        self.correction_term.double()
    # ============================================
    def forward(self, x): 

        MLdHdq = self.correction_term(x)
        return MLdHdq