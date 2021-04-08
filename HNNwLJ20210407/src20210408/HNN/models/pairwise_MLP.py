import torch.nn as nn

class pairwise_MLP(nn.Module):

    def __init__(self, MLP_input = 5, MLP_nhidden = 32):
        super(pairwise_MLP, self).__init__()

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
        self.correction_term.double()
    # ============================================
    def forward(self, x): 

        MLdHdq = self.correction_term(x)
        return MLdHdq
