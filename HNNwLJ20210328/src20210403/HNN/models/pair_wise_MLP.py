import torch
import torch.nn as nn
from parameters.ML_parameters import ML_parameters

class pair_wise_MLP(nn.Module):

    def __init__(self):
        super(pair_wise_MLP, self).__init__()

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


    def forward(self, x): # data -> del_list ( del_qx, del_qy, del_px, del_py, t )

        MLdHdq = self.correction_term(x.float())
        # print('mldhdq', MLdHdq)

        # print('w 1',self.correction_term[0].weight)
        # print('w 2', self.correction_term[2].weight)
        # print('ML output',MLdHdq_)

        return MLdHdq
