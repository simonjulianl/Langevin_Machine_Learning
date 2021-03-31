import torch
from MC_parameters import MC_parameters
from MD_parameters import MD_parameters
from ML_parameters import ML_parameters
from hamiltonian import hamiltonian
from hamiltonian.lennard_jones import lennard_jones
from hamiltonian.kinetic_energy import kinetic_energy
from utils.get_paired_distance_indices import get_paired_distance_indices
import time

class pair_wise_HNN(hamiltonian):

    ''' pair_wise_HNN class to learn dHdq and then combine with nodHdq '''

    _obj_count = 0

    def __init__(self, pair_wise_MLP):

        pair_wise_HNN._obj_count += 1
        assert (pair_wise_HNN._obj_count == 1),type(self).__name__ + " has more than one object"

        super().__init__()
        self.network = pair_wise_MLP

        # # append term to calculate dHdq
        super().append(lennard_jones())
        super().append(kinetic_energy(MD_parameters.mass))

    def train(self):
        self.network.train()  # pytorch network for training

    def eval(self):   # pytorch network for eval
        self.network.eval()

    def dHdq(self, phase_space):

        ''' function to calculate dHdq = noML_dHdq + residual ML_dHdq

        Parameters
        ----------
        data : torch.tensor
                shape is [(nsamples x nparticle x nparticle-1), (del_qx, del_qy, del_px, del_py, tau )]
        noML_dHdq : torch.tensor (cpu)
                shape is [nparticle, DIM]
                use .to(device) to move a tensor to device
        predict : torch.tensor (gpu)
                shape is [nparticle, DIM]
        Returns
        ----------
        corrected_dHdq : torch.tensor
                shape is [nparticle, DIM]

        '''

        q_list = phase_space.get_q()
        p_list = phase_space.get_p()

        noML_dHdq = super().dHdq(phase_space)
        noML_dHdq = noML_dHdq.squeeze(dim=0)

        phase_space.set_q(q_list)
        phase_space.set_p(p_list)

        data = self.phase_space2data(phase_space, MD_parameters.tau_long)
        predict = self.network(data, MC_parameters.nparticle, MC_parameters.DIM)

        corrected_dHdq = noML_dHdq.to(ML_parameters.device) + predict

        return corrected_dHdq

    def phase_space2data(self, phase_space, tau_cur):

        ''' function to prepare input in nn
        Parameters
        ----------
        tau_cur : float
                large time step for input in neural network
        nsamples : int
                1 because load one sample during training

        Returns
        ----------
        input in neural network
        shape is [(nsamples x nparticle x nparticle-1), (del_qx, del_qy, del_px, del_py, tau )]

        '''

        q_list = phase_space.get_q()
        p_list = phase_space.get_p()

        # print('phase_space2data',q_list.shape, p_list.shape)
        nsamples, nparticle, DIM = q_list.shape

        delta_init_q = torch.zeros((nsamples, nparticle, (nparticle - 1), DIM)).to(ML_parameters.device)
        delta_init_p = torch.zeros((nsamples, nparticle, (nparticle - 1), DIM)).to(ML_parameters.device)

        # load one sample during training
        for z in range(nsamples):

            delta_init_q_ = self.delta_state(q_list[z:z+1], nparticle, DIM)
            delta_init_p_ = self.delta_state(p_list[z:z+1], nparticle, DIM)

            delta_init_q[z] = delta_init_q_
            delta_init_p[z] = delta_init_p_

        # to add paired data, tau reshape
        tau = torch.tensor([tau_cur] * nparticle * (nparticle - 1)).to(ML_parameters.device)
        tau = tau.reshape(-1, nparticle, (nparticle - 1), 1)  # broadcasting

        paired_data_ = torch.cat((delta_init_q, delta_init_p), dim=-1)  # [nsamples, nparticle, (nparticle-1), (del_qx, del_qy, del_px, del_py)]
        paired_data = torch.cat((paired_data_, tau), dim=-1)  # [nsamples, nparticle, (nparticle-1), (del_qx, del_qy, del_px, del_py, tau )]
        paired_data = paired_data.reshape(-1, paired_data.shape[3])  # [(nsamples x nparticle x nparticle-1), (del_qx, del_qy, del_px, del_py, tau )]
        print('paired_data', paired_data.shape)

        return paired_data

    def delta_state(self, state_list, nparticle, DIM):

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
        dstate : shape is [nsamples, nparticle, nparticle, DIM]
                distance of q or p btw two particles
        dstate_flatten : shape is [nsamples x nparticle x (nparticle - 1) x DIM]
        dstate_reshape : shape is [nsamples, nparticle, (nparticle - 1), DIM]

        Returns
        ----------
        dstate_reshape :  obtain dq or dp of non-zero indices
        '''

        state_len = state_list.shape[1]
        state0 = torch.unsqueeze(state_list, dim=1)
        statem = torch.repeat_interleave(state0, state_len, dim=1)

        statet = statem.permute(get_paired_distance_indices.permute_order)

        dstate = statet - statem

        dstate_reduced_index = get_paired_distance_indices.get_indices(dstate.shape)
        dstate_flatten = get_paired_distance_indices.reduce(dstate, dstate_reduced_index)

        dstate_reshape = dstate_flatten.reshape((state_list.shape[0], nparticle, nparticle - 1, DIM))

        return dstate_reshape
