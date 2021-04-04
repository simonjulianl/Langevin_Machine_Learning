import torch
from parameters.ML_parameters           import ML_parameters
from hamiltonian.hamiltonian            import hamiltonian
from hamiltonian.lennard_jones          import lennard_jones
from hamiltonian.kinetic_energy         import kinetic_energy
from utils.get_paired_distance_indices  import get_paired_distance_indices


class pair_wise_HNN(hamiltonian):

    ''' pair_wise_HNN class to learn dHdq and then combine with nodHdq '''

    _obj_count = 0

    def __init__(self, pair_wise_MLP, tau_cur):

        pair_wise_HNN._obj_count += 1
        assert (pair_wise_HNN._obj_count == 1),type(self).__name__ + " has more than one object"

        super().__init__()
        self.network = pair_wise_MLP

        # # append term to calculate dHdq
        super().append(lennard_jones())
        super().append(kinetic_energy())

        self.tau_cur = tau_cur

    def train(self):
        self.network.train()  # pytorch network for training

    def eval(self):   # pytorch network for eval
        self.network.eval()

    def dHdq(self, phase_space):

        ''' function to calculate dHdq = noML_dHdq + residual ML_dHdq

        Parameters
        ----------
        x : torch.tensor
                shape is [(nsamples x nparticle x nparticle), (del_qx, del_qy, del_px, del_py, tau )]
        noML_dHdq : torch.tensor (cpu)
                shape is [nsamples, nparticle, DIM]
                use .to(device) to move a tensor to device
        predict : torch.tensor (gpu)
                shape is [nsamples, nparticle, DIM]

        Returns
        ----------
        corrected_dHdq : torch.tensor
                shape is [nsamples, nparticle, DIM]

        '''

        q_list = phase_space.get_q()
        p_list = phase_space.get_p()

        nsamples, nparticle, DIM = q_list.shape

        # print('noML dhdq')
        noML_dHdq = super().dHdq(phase_space)
        # print('noML dhdq', noML_dHdq)

        phase_space.set_q(q_list)
        phase_space.set_p(p_list)

        # print('ML dhdq')
        dqdp = self.make_dqdp(phase_space)
        tau = self.make_proper_tau_shape(self.tau_cur * 0.5, nsamples, nparticle)

        # shape is [nsamples, nparticle, nparticle, 5]; here 5 is (dq_x, dq_y, dp_x, dp_y, tau)
        x = torch.cat((dqdp, tau), dim=-1)

        x = x.reshape(-1, x.shape[3])

        output = self.network(x)
        output = output.reshape( nsamples, nparticle, nparticle, DIM)

        ML_dHdq = torch.sum(output, dim=2) # e.g) a,b,c three particles;  sum fa = fab + fac
        # print('ML dhdq', ML_dHdq)

        corrected_dHdq = noML_dHdq + ML_dHdq

        return corrected_dHdq

    def make_proper_tau_shape(self, tau_cur, nsamples, nparticle):

        ''' function to make proper shape for tau

        Parameters
        ----------
        tau_cur : float
                half long time step for input in neural network

        Returns
        ----------
        tau : torch.tensor
        shape is [ nsamples, nparticle, nparticle, 1]

        '''

        tau = torch.tensor([tau_cur] * nsamples * nparticle * nparticle, dtype=torch.float64)
        tau = tau.reshape(-1, nparticle, nparticle, 1)  # broadcasting

        return tau

    def make_dqdp(self, phase_space):

        ''' function to make dq and dp for feeding into nn

        Returns
        ----------
        dqdp : shape is [ nsamples, nparticle, nparticle, 4]
        here 4 is (dq_x, dq_y, dp_x, dp_y )
        '''

        q_list = phase_space.get_q()
        p_list = phase_space.get_p()

        dq = self.delta_state(q_list)
        dp = self.delta_state(p_list)

        dqdp = torch.cat((dq, dp), dim=-1)

        return dqdp


    def delta_state(self, state_list):

        ''' function to calculate distance of q or p between two particles

        Parameters
        ----------
        state_list : torch.tensor
                shape is [nsamples, nparticle, DIM]
        state_len : nparticle
        state0 : shape is [nsamples, 1, nparticle, DIM]
                new tensor with a dimension of size one inserted at the specified position (dim=1)
        statem : shape is [nsamples, nparticle, nparticle, DIM]
                repeated tensor which has the same shape as state0 along with dim=1
        statet : shape is [nsamples, nparticle, nparticle, DIM]
                permutes the order of the axes of a tensor

        Returns
        ----------
        dstate : shape is [nsamples, nparticle, nparticle, DIM]
                distance of q or p btw two particles
        '''

        state_len = state_list.shape[1]
        state0 = torch.unsqueeze(state_list, dim=1)
        statem = torch.repeat_interleave(state0, state_len, dim=1)

        statet = statem.permute(get_paired_distance_indices.permute_order)

        dstate = statet - statem

        return dstate
