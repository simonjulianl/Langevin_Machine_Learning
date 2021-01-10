import torch
import torch.nn as nn
from linear_integrator import linear_integrator

class pair_wise_HNN:

    def __init__(self, hamiltonian, network, **kwargs):
        self.network = network
        self.noML_hamiltonian = hamiltonian
        self._state = kwargs

    def train(self):
        self.network.train() # pytorch network for training

    def eval(self):
        self.network.eval()

    # def loss(self, prediction, label):
    #
    #     q_list_predict, p_list_predict = prediction
    #     q_list_label, p_list_label = label
    #
    #     _reduction = 'sum'
    #     criterion = nn.MSELoss(reduction = _reduction)
    #     loss = criterion(q_list_predict, q_list_label) + criterion(p_list_predict, p_list_label)
    #
    #     return loss

    def dHdq(self, phase_space, pb):
        data = self.phase_space2data(phase_space,pb)
        data = data.requires_grad_(True)
        print('dHdq',data)
        predict = self.network(data, self._state['particle'], self._state['DIM'])
        print('pred',predict)
        noML_dHdq = self.noML_hamiltonian.dHdq(phase_space,pb)
        print('noML_dHdq',noML_dHdq)
        corrected_dHdq = noML_dHdq + predict  # in linear_vv code, it calculates grad potential.
        print('correction',corrected_dHdq)
        return corrected_dHdq

    def phase_spacedata(self,phase_space, pb):

        print('phase_spacedata')
        print(phase_space.get_q())
        print(phase_space.get_p())
        _state = linear_integrator(**self._state).integrate(self.noML_hamiltonian)

        q_list_label = phase_space.get_q()
        p_list_label = phase_space.get_p()

        return  (q_list_label, p_list_label)

    def phase_space2data(self,phase_space, pb):

         print('phase_space2data')
         q_list = phase_space.get_q()
         p_list = phase_space.get_p()


         print(q_list, p_list)
         N, N_particle, DIM = q_list.shape
         print(q_list.shape)

         delta_init_q = torch.zeros((N, N_particle, (N_particle - 1), DIM))
         delta_init_p = torch.zeros((N, N_particle, (N_particle - 1), DIM))

         for z in range(N):

             delta_init_q_, _ = pb.paired_distance_reduced(q_list[z] / self._state['BoxSize'], N_particle, DIM)  # reduce distance for pb
             delta_init_q_ = delta_init_q_ * self._state['BoxSize'] # not reduce distance

             delta_init_p_, _ = pb.paired_distance_reduced(p_list[z] / self._state['BoxSize'], N_particle, DIM)  # reduce distance for pb
             delta_init_p_ = delta_init_p_ * self._state['BoxSize']

             delta_init_q[z] = delta_init_q_
             delta_init_p[z] = delta_init_p_
             # print('delta_init_q',delta_init_q_)
             # print('delta_init_q',delta_init_q_.shape)
             # print('delta_init_p',delta_init_p_)
             # print('delta_init_p', delta_init_p_.shape)

             # delta_q_x, delta_q_y, t
             # for i in range(N_particle):
             #     x = 0  # all index case i=j and i != j
             #     for j in range(N_particle):
             #         if i != j:
             #             # print(i,j)
             #             # print(delta_init_q_[i,j,:])
             #             delta_init_q[z][i][x] = delta_init_q_[i, j, :]
             #             delta_init_p[z][i][x] = delta_init_p_[i, j, :]
             #
             #             x = x + 1

         print('delta_init')
         print(delta_init_q)
         print(delta_init_p)

         # tau : #this is big time step to be trained
         # to add paired data array
         tau = torch.tensor([self._state['tau'] * self._state['iterations']] * N_particle * (N_particle - 1))
         tau = tau.reshape(-1, N_particle, (N_particle - 1), 1)  # broadcasting
         # print('tau', tau.shape)
         paired_data_ = torch.cat((delta_init_q, delta_init_p), dim=-1)  # N (nsamples) x N_particle x (N_particle-1) x (del_qx, del_qy, del_px, del_py)
         # print(paired_data_)
         # print(paired_data_.shape)
         paired_data = torch.cat((paired_data_, tau), dim=-1)  # nsamples x N_particle x (N_particle-1) x  (del_qx, del_qy, del_px, del_py, tau )
         paired_data = paired_data.reshape(-1, paired_data.shape[3]) # (nsamples x N_particle) x (N_particle-1) x  (del_qx, del_qy, del_px, del_py, tau )

         print('=== input data for ML : del_qx del_qy del_px del_py tau ===')
         print(paired_data)
         print(paired_data.shape)

         return paired_data