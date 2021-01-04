import torch
import torch.nn as nn
import numpy as np

class pair_wise_MLP(nn.Module):

    def __init__(self, n_input, n_hidden):
        '''
        VV : velocity verlet

        please check MLP2H_Separable_Hamil_LF.py for full documentation

        '''
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

    def forward(self, q_list, p_list, del_list, batch_idx, **kwargs):
        '''
        Forward pass using velocity verlet, only for DIM = 1

        Parameters
        ----------
        q_list : torch.tensor of N X 1
            tensor of position
        p_list : torch.tensor of N X 1
            tensor of momentum

        time_step : float
            as described by dataset

        Returns
        -------
        q_list,p_list : torch.tensor
            torch.tensor of (q_next,p_next) of N X 2

        please check MLP2H_Separable_Hamil_LF.py for full documentation
        '''


        print(kwargs)
        print('MLP',del_list)
        Hamiltonian = kwargs['hamiltonian']

        kwargs['phase_space'].set_q(q_list)
        kwargs['phase_space'].set_p(p_list)

        p_list_dummy = np.zeros(p_list.shape)  # to prevent KE from being integrated
        kwargs['phase_space'].set_p(p_list_dummy)

        noMLdHdq = Hamiltonian.dHdq(kwargs['phase_space'], kwargs['pb_q'])
        noMLdHdq = torch.from_numpy(noMLdHdq)

        print(del_list)
        quit()
        MLdHdq = self.correction_term(del_list)

        p_list = p_list + (kwargs['tau'] * kwargs['iterations']) / 2 * (- (noMLdHdq + MLdHdq) ) #this is big time step to be trained
        # we need to use prediction again instead of using p_list because we need correction

        q_list = q_list + (kwargs['tau'] * kwargs['iterations']) * p_list  # dq/dt = dK/dp = p

        q_list = q_list.detach().cpu().numpy()
        kwargs['pb_q'].adjust_real(q_list, kwargs['BoxSize'])
        kwargs['phase_space'].set_q(q_list)

        kwargs['pb_q'].debug_pbc(q_list, kwargs['BoxSize'])
        q_list = torch.from_numpy(q_list)

        noMLdHdq = Hamiltonian.dHdq(kwargs['phase_space'], kwargs['pb_q'])
        noMLdHdq = torch.from_numpy(noMLdHdq)
        MLdHdq = self.correction_term(del_list)

        p_list = p_list + (kwargs['tau'] * kwargs['iterations']) / 2 * (- (noMLdHdq + MLdHdq) )  # dp/dt

        print('vv',q_list[0][batch_idx],p_list[0][batch_idx])
        return (q_list[0][batch_idx], p_list[0][batch_idx])