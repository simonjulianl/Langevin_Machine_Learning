import torch

class pair_wise_HNN:

    def __init__(self,hamiltonian,network):
        self.network = network
        self.noML_hamiltonian = hamiltonian

    def train(self):
        self.network.train() # pytorch network for training

    def eval(self):
        self.network.eval()

    def dHdq(self, phase_space, pb):
        data = self.phase_space2data(phase_space,pb)
        predict = self.network(data)
        noML_force = self.noML_hamiltonian.dHdq(phase_space,pb)

        corrected_force = noML_force + predict

        return corrected_force

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

             delta_init_q_, _ = self.pb.paired_distance_reduced(q_list[z] / self._state['BoxSize'])  # reduce distance
             delta_init_q_ = delta_init_q_ * self._state['BoxSize']
             delta_init_p_, _ = self.pb.paired_distance_reduced(p_list[z] / self._state['BoxSize'])  # reduce distance
             delta_init_p_ = delta_init_p_ * self._state['BoxSize']

             # print('delta_init_q',delta_init_q_)
             # print('delta_init_q',delta_init_q_.shape)
             # print('delta_init_p',delta_init_p_)
             # print('delta_init_p', delta_init_p_.shape)

             # delta_q_x, delta_q_y, t
             for i in range(N_particle):
                 x = 0  # all index case i=j and i != j
                 for j in range(N_particle):
                     if i != j:
                         # print(i,j)
                         # print(delta_init_q_[i,j,:])
                         delta_init_q[z][i][x] = delta_init_q_[i, j, :]
                         delta_init_p[z][i][x] = delta_init_p_[i, j, :]

                         x = x + 1

         # print('delta_init')
         # print(delta_init_q)
         # print(delta_init_p)

         # tau : #this is big time step to be trained
         # to add paired data array
         tau = torch.array([self._state['tau'] * self._state['iterations']] * N_particle * (N_particle - 1))
         tau = tau.reshape(-1, N_particle, (N_particle - 1), 1)  # broadcasting
         print('tau', tau.shape)
         # print('concat')
         paired_data_ = torch.concatenate((delta_init_q, delta_init_p),
                                       axis=-1)  # N (nsamples) x N_particle x (N_particle-1) x (del_qx, del_qy, del_px, del_py)
         # print(paired_data_)
         # print(paired_data_.shape)
         paired_data = torch.concatenate((paired_data_, tau),
                                      axis=-1)  # nsamples x N_particle x (N_particle-1) x  (del_qx, del_qy, del_px, del_py, tau )
         paired_data = paired_data.reshape(-1, paired_data.shape[3])  # (nsamples x N_particle) x (N_particle-1) x  (del_qx, del_qy, del_px, del_py, tau )

         print('=== input data for ML : del_qx del_qy del_px del_py tau ===')
         print(paired_data)
         print(paired_data.shape)

         return paired_data