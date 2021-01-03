import torch
import torch.nn as nn
import numpy as np

class pair_wise_MLP(nn.Module):

    def __init__(self, n_input, n_hidden):
        '''
        VV : velocity verlet

        please check MLP2H_Separable_Hamil_LF.py for full documentation

        '''
        super(pair_wise_MLP, self).__init__()
        self.correction_term = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, 2)
        )

    def forward(self, del_list):

        MLdHdq = self.correction_term(del_list)

        MLdHdq = MLdHdq.sum(0)

        return MLdHdq