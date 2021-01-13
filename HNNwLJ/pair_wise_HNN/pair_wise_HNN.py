import torch

class pair_wise_HNN:

    def __init__(self, NoML_hamiltonian, network, **kwargs):
        self.network = network
        self.noML_hamiltonian = NoML_hamiltonian
        self._state = kwargs

    def train(self):
        self.network.train()  # pytorch network for training

    def eval(self):
        self.network.eval()

    def dHdq(self, phase_space, pb):

        q_list = phase_space.get_q()
        p_list = phase_space.get_p()

        # print('===== data for noML dHdq =====')
        # print(q_list,p_list)

        noML_dHdq = self.noML_hamiltonian.dHdq(phase_space, pb)

        phase_space.set_q(q_list)
        phase_space.set_p(p_list)

        # print('===== data for preparing ML input =====')
        data = self.phase_space2data(phase_space)
        data = data.requires_grad_(True)
        # print('=== input for ML : del_qx del_qy del_px del_py tau ===')
        # print(data)
        # print(data.shape)

        predict = self.network(data, self._state['nparticle'], self._state['DIM'])

        corrected_dHdq = noML_dHdq + predict  # in linear_vv code, it calculates grad potential.

        return corrected_dHdq

    def phase_space2data(self, phase_space):

        q_list = phase_space.get_q()
        p_list = phase_space.get_p()

        # print('phase_space2data',q_list.shape, p_list.shape)
        nsamples, nparticle, DIM = q_list.shape

        delta_init_q = torch.zeros((nsamples, nparticle, (nparticle - 1), DIM))
        delta_init_p = torch.zeros((nsamples, nparticle, (nparticle - 1), DIM))

        for z in range(nsamples):

            delta_init_q_ = self.delta_qorp(q_list[z], nparticle, DIM)
            delta_init_p_ = self.delta_qorp(p_list[z], nparticle, DIM)

            delta_init_q[z] = delta_init_q_
            delta_init_p[z] = delta_init_p_

        # tau : #this is big time step to be trained
        # to add paired data array, reshape
        tau = torch.tensor([self._state['tau_cur']] * nparticle * (nparticle - 1))
        tau = tau.reshape(-1, nparticle, (nparticle - 1), 1)  # broadcasting
        # print(tau)

        # print(delta_init_q.shape,delta_init_p.shape)

        paired_data_ = torch.cat((delta_init_q, delta_init_p), dim=-1)  # N (nsamples) x N_particle x (N_particle-1) x (del_qx, del_qy, del_px, del_py)
        paired_data = torch.cat((paired_data_, tau), dim=-1)  # nsamples x N_particle x (N_particle-1) x  (del_qx, del_qy, del_px, del_py, tau )
        paired_data = paired_data.reshape(-1, paired_data.shape[3])  # (nsamples x N_particle) x (N_particle-1) x  (del_qx, del_qy, del_px, del_py, tau )

        return paired_data

    def delta_qorp(self, qorp_list, nparticle, DIM):

        qorp_len = qorp_list.shape[0]
        qorp0 = torch.unsqueeze(qorp_list, dim=0)
        qorpm = torch.repeat_interleave(qorp0, qorp_len, dim=0)

        qorpt = qorpm.permute(1,0,2)

        dqorp_ = qorpt - qorpm

        dqorp = torch.zeros((nparticle,(nparticle - 1),DIM))

        for i in range(nparticle):
            x=0
            for j in range(nparticle):
                if i != j:
                    dqorp[i][x] = dqorp_[i,j,:]

                    x = x + 1

        return dqorp

