import torch
import torch.nn as nn
import numpy as np

class pair_wise_MLP(nn.Module):

    def __init__(self, n_input, n_hidden):

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

    def forward(self, data): # data -> del_list ( del_qx, del_qy, del_px, del_py, t )

        MLdHdq = self.correction_term(data)

        MLdHdq = MLdHdq.sum(0)  # ex) a,b,c three particles;  sum Fa = Fab + Fac

        return MLdHdq