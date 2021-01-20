import torch
import torch.nn as nn

class pair_wise_zero:

    def __init__(self, n_input, n_hidden):
        print('zero nn, do nothing')


    def forward(self,data, nparticle, DIM): # data -> del_list ( del_qx, del_qy, del_px, del_py, t )

        zerodHdq = torch.zeros(nparticle,DIM)

        return zerodHdq