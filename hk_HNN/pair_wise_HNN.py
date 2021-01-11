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

        print('pair_wise',q_list,p_list)
        noML_dHdq = self.noML_hamiltonian.dHdq(phase_space, pb)
        print('noML_dHdq', noML_dHdq)

        phase_space.set_q(q_list)
        phase_space.set_p(p_list)

        data = self.phase_space2data(phase_space, pb)
        data = data.requires_grad_(True)
        print('input for ML', data)

        predict = self.network(data, self._state['particle'], self._state['DIM'])
        print('pred', predict)

        corrected_dHdq = noML_dHdq + predict  # in linear_vv code, it calculates grad potential.
        print('corrected_dHdq', corrected_dHdq)

        return corrected_dHdq

    def phase_space2data(self, phase_space, pb):

        print('phase_space2data')
        q_list = phase_space.get_q()
        p_list = phase_space.get_p()

        print(q_list, p_list)
        N, N_particle, DIM = q_list.shape

        delta_init_q = torch.zeros((N, N_particle, (N_particle - 1), DIM))
        delta_init_p = torch.zeros((N, N_particle, (N_particle - 1), DIM))

        for z in range(N):

            delta_init_q_ = self.delta_qp(q_list[z], N_particle, DIM)
            print('delta q ',delta_init_q_)
            delta_init_p_ = self.delta_qp(p_list[z], N_particle, DIM)
            print('delta p ',delta_init_p_)

            delta_init_q[z] = delta_init_q_
            delta_init_p[z] = delta_init_p_

        print('delta_init')
        print(delta_init_q)
        print(delta_init_p)

        # tau : #this is big time step to be trained
        # to add paired data array, reshape
        tau = torch.tensor([self._state['tau']] * N_particle * (N_particle - 1))
        tau = tau.reshape(-1, N_particle, (N_particle - 1), 1)  # broadcasting
        # print(tau)

        paired_data_ = torch.cat((delta_init_q, delta_init_p), dim=-1)  # N (nsamples) x N_particle x (N_particle-1) x (del_qx, del_qy, del_px, del_py)
        paired_data = torch.cat((paired_data_, tau), dim=-1)  # nsamples x N_particle x (N_particle-1) x  (del_qx, del_qy, del_px, del_py, tau )
        paired_data = paired_data.reshape(-1, paired_data.shape[3])  # (nsamples x N_particle) x (N_particle-1) x  (del_qx, del_qy, del_px, del_py, tau )

        print('=== input data for ML : del_qx del_qy del_px del_py tau ===')
        print(paired_data)
        print(paired_data.shape)

        return paired_data

    def delta_qp(self, qp_list, N_particle, DIM):

        qp_len = qp_list.shape[0]
        qp0 = torch.unsqueeze(qp_list,dim=0)
        qpm = torch.repeat_interleave(qp0,qp_len,dim=0)
        #print(qpm)
        qpt = qpm.permute(1,0,2)
        #print(qpt)
        dqp_ = qpt - qpm

        dqp = torch.zeros((N_particle,(N_particle-1),DIM))

        for i in range(N_particle):
            x=0
            for j in range(N_particle):
                if i != j:
                    dqp[i][x] = dqp_[i,j,:]

                    x = x + 1

        return dqp

