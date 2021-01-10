import torch
import torch.nn as nn

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

    def forward(self,data, N_particle, DIM): # data -> del_list ( del_qx, del_qy, del_px, del_py, t )

        MLdHdq_ = self.correction_term(data)
        # print('MLP',MLdHdq_)
        MLdHdq_ = MLdHdq_.reshape(N_particle, N_particle-1, DIM)  # N_particle, N_particle-1, DIM
        # print('MLP',MLdHdq_)
        MLdHdq = torch.sum(MLdHdq_, dim=1) # ex) a,b,c three particles;  sum Fa = Fab + Fac
        # print('MLP',MLdHdq)
        return MLdHdq