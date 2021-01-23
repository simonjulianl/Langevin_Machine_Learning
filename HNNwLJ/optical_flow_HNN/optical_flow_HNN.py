import torch

class optical_flow_HNN:

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
        # print('noML_dHdq',noML_dHdq)

        phase_space.set_q(q_list)
        phase_space.set_p(p_list)

        data = self.phase_space2data(phase_space)

        # print('=== input for ML : del_qx del_qy del_px del_py tau ===')
        # print(data)
        # print(data.shape)

        predict = self.network(data, self._state['nparticle'], self._state['DIM'])
        # print('nn output',predict)

        corrected_dHdq = noML_dHdq.to(self._state['_device']) + predict  # in linear_vv code, it calculates grad potential.
        # print('noML_dHdq diff',noML_dHdq.to(self._state['_device']) -corrected_dHdq)
        # print('corrected_dHdq',corrected_dHdq)

        return corrected_dHdq

    def phase_space2data(self, phase_space):

        q_list = phase_space.get_q()
        p_list = phase_space.get_p()

