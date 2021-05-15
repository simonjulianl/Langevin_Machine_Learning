import torch.nn as nn
import torch

class fields_MLP(nn.Module):

    def __init__(self, MLP_input, MLP_nhidden):
        super(fields_MLP, self).__init__()

        self.correction_term = nn.Sequential(
            nn.Linear(MLP_input, MLP_nhidden),
            nn.Tanh(),
            nn.Linear(MLP_nhidden, MLP_nhidden),
            nn.Tanh(),
            nn.Linear(MLP_nhidden, 16),
            nn.Tanh(),
            nn.Linear(16, 2)
        )
        self.correction_term.double()
        self.correction_term.apply(self.init_weights)

        print('pairwise_MLP initialized : ',MLP_input,'->',MLP_nhidden,'->',MLP_nhidden,'-> 16 -> 2')
    # ============================================
    def init_weights(self,layer):
        if type(layer) == nn.Linear:
            nn.init.xavier_normal_(layer.weight,gain=0.1) # SJ make weights small
            layer.bias.data.fill_(0.0)
    # ============================================
    def forward(self, x):

        # x.shape is [nsamples*nparticle, grids18 + grids18]

        MLdHdq = self.correction_term(x)
        # MLdHdq.shape is [nsamples*nparticle,2]

        # print('w 1',self.correction_term[0].weight)
        # print('w 2', self.correction_term[2].weight)
        # print('w 3', self.correction_term[4].weight)
        # print('ML output',MLdHdq)

        return MLdHdq
