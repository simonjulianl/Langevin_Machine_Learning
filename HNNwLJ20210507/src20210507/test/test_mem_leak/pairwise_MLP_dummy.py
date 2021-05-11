import torch.nn as nn
import torch

class pairwise_MLP_dummy(nn.Module):

    def __init__(self, MLP_input, MLP_nhidden):
        super(pairwise_MLP_dummy, self).__init__()

        self.correction_term = nn.Sequential(
            nn.Linear(MLP_input,2)
        )
        self.correction_term.double()

        print('pairwise_MLP dummy initialized ')
    # ============================================
    def forward(self, x):

        # x.shape is [nsamples * nparticle * nparticle, 5]
        nsamplesnaprticle, _ = x.shape
        MLdHdq = self.correction_term(x)
        #MLdHdq = torch.zeros((nsamplesnaprticle,2))
        # MLdHdq.shape is [nsamples * nparticle * nparticle, 2]

        # print('w 1',self.correction_term[0].weight)
        # print('w 2', self.correction_term[2].weight)
        # print('w 3', self.correction_term[4].weight)
        # print('ML output',MLdHdq)

        return MLdHdq
