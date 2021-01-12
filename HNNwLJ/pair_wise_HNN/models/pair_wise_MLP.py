import torch
import torch.nn as nn

class pair_wise_MLP(nn.Module):

    def __init__(self, n_input, n_hidden):

        super(pair_wise_MLP, self).__init__()
        self.correction_term = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, 2)
        )

    def forward(self,data, nparticle, DIM): # data -> del_list ( del_qx, del_qy, del_px, del_py, t )

        MLdHdq_ = self.correction_term(data)

        # print('w 1',self.correction_term[0].weight)
        # print('w 2', self.correction_term[2].weight)
        # print('ML output',MLdHdq_)

        MLdHdq_ = MLdHdq_.reshape(nparticle, nparticle - 1, DIM)  # N_particle, N_particle-1, DIM
        # print('output reshape',MLdHdq_)
        MLdHdq = torch.sum(MLdHdq_, dim=1) # ex) a,b,c three particles;  sum Fa = Fab + Fac
        # print('output sum',MLdHdq)
        return MLdHdq