import torch
import torch.nn as nn
from pb import pb
from phase_space import phase_space

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

    def forward(self, data):

        output = self.correction_term(data)

        return output


if __name__ == '__main__':

    q_list = [[[3,2],[2.2,1.21]]]
    p_list = [[[0.,0.],[0.,0.]]]

    q_list_tensor, p_list_tensor = torch.tensor([q_list,p_list])

    pb = pb()
    phase_space = phase_space()

    state = {

        }




