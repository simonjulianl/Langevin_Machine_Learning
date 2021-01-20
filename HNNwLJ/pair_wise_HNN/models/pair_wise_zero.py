import torch
import torch.nn as nn

class pair_wise_zero(nn.Module):

    def __init__(self, n_input, n_hidden):
        super(pair_wise_zero, self).__init__()
        self.correction_term = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.Linear(n_hidden, 2)
        )

        print('zero nn, do nothing')


    def forward(self,data, nparticle, DIM): # data -> del_list ( del_qx, del_qy, del_px, del_py, t )

        unused = self.correction_term(data)
        zerodHdq_ = (unused-unused)
        zerodHdq_ = zerodHdq_.reshape(nparticle, nparticle - 1, DIM)
        zerodHdq = torch.sum(zerodHdq_, dim=1)

        return zerodHdq