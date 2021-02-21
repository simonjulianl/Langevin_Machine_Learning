import torch
import torch.nn as nn
from MD_parameters import MD_parameters
from ML_parameters import ML_parameters

class pair_wise_zero(nn.Module):

    def __init__(self):
        super(pair_wise_zero, self).__init__()
        MLP_input = ML_parameters.MLP_input
        MLP_nhidden = ML_parameters.MLP_nhidden

        self.correction_term = nn.Sequential(
            nn.Linear(MLP_input, MLP_nhidden),
            nn.Linear(MLP_nhidden, 2)
        )

        print('zero nn, do nothing')


    def forward(self,data, nparticle, DIM): # data -> del_list ( del_qx, del_qy, del_px, del_py, t )

        unused = self.correction_term(data)
        zerodHdq_ = (unused-unused)
        zerodHdq_ = zerodHdq_.reshape(MD_parameters.nsamples_batch, nparticle, nparticle - 1, DIM)
        zerodHdq = torch.sum(zerodHdq_, dim=2)

        return zerodHdq