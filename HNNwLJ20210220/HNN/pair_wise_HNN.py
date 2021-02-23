import torch
from MD_parameters import MD_parameters
from ML_parameters import ML_parameters
from hamiltonian import hamiltonian
from hamiltonian.lennard_jones import lennard_jones
from hamiltonian.kinetic_energy import kinetic_energy
from utils.paired_distance_reduce import paired_distance_reduce
import time
class pair_wise_HNN(hamiltonian):

    def helper(self = None):
        '''print the common parameters helper'''
        for parent in pair_wise_HNN.__bases__:
            print(help(parent))

    _obj_count = 0

    def __init__(self, network):

        pair_wise_HNN._obj_count += 1
        assert (pair_wise_HNN._obj_count == 1),type(self).__name__ + " has more than one object"

        super().__init__()
        self.network = network

        # # append term to calculate dHdq
        super().append(lennard_jones())
        super().append(kinetic_energy(MD_parameters.mass))

    def train(self):
        self.network.train()  # pytorch network for training

    def eval(self):
        self.network.eval()

    def dHdq(self, phase_space):

        start_dhdq = time.time()
        # print('call pair_wise_HNN class')
        q_list = phase_space.get_q()
        p_list = phase_space.get_p()

        # print('===== data for noML dHdq =====')
        # print(q_list,p_list)

        start_noML = time.time()
        noML_dHdq = super().dHdq(phase_space)
        end_noML = time.time()

        self.noML_time = end_noML - start_noML
        #print('time for noML', end_noML - start_noML)
        # print('noML_dHdq',noML_dHdq.device)

        phase_space.set_q(q_list)
        phase_space.set_p(p_list)

        start_prep_data = time.time()
        data = self.phase_space2data(phase_space, MD_parameters.tau_long)
        end_prep_data = time.time()

        self.prep_data_time = end_prep_data - start_prep_data
        #print('time for prepare data ML', end_prep_data - start_prep_data)

        # print('=== input for ML : del_qx del_qy del_px del_py tau ===')
        # print(data)

        start_ML = time.time()
        predict = self.network(data, MD_parameters.nparticle, MD_parameters.DIM)
        end_ML = time.time()
        self.ML_time = end_ML - start_ML
        #print('time for ML', end_ML - start_ML)
        # print('nn output',predict)
        # print(predict.device)

        start_corrected = time.time()
        corrected_dHdq = noML_dHdq.to(ML_parameters.device) + predict  # in linear_vv code, it calculates grad potential.
        end_corrected = time.time()

        self.corrected_time = end_corrected - start_corrected
        end_dhdq = time.time()
        self.dhdq_time = end_dhdq - start_dhdq
        # print('noML_dHdq diff',noML_dHdq.to(self._state['_device']) -corrected_dHdq)
        # print('corrected_dHdq',corrected_dHdq)

        return corrected_dHdq

    def phase_space2data(self, phase_space, tau_cur):

        q_list = phase_space.get_q()
        p_list = phase_space.get_p()

        # print('phase_space2data',q_list.shape, p_list.shape)
        nsamples, nparticle, DIM = q_list.shape

        delta_init_q = torch.zeros((nsamples, nparticle, (nparticle - 1), DIM)).to(ML_parameters.device)
        delta_init_p = torch.zeros((nsamples, nparticle, (nparticle - 1), DIM)).to(ML_parameters.device)

        for z in range(nsamples):

            delta_init_q_ = self.delta_qorp(q_list[z], nparticle, DIM)
            delta_init_p_ = self.delta_qorp(p_list[z], nparticle, DIM)

            delta_init_q[z] = delta_init_q_
            delta_init_p[z] = delta_init_p_

        # tau : #this is big time step to be trained
        # to add paired data array, reshape
        tau = torch.tensor([tau_cur] * nparticle * (nparticle - 1)).to(ML_parameters.device)
        tau = tau.reshape(-1, nparticle, (nparticle - 1), 1)  # broadcasting

        paired_data_ = torch.cat((delta_init_q, delta_init_p), dim=-1)  # nsamples x nparticle x (nparticle-1) x (del_qx, del_qy, del_px, del_py)
        paired_data = torch.cat((paired_data_, tau), dim=-1)  # nsamples x nparticle x (nparticle-1) x  (del_qx, del_qy, del_px, del_py, tau )
        paired_data = paired_data.reshape(-1, paired_data.shape[3])  # (nsamples x nparticle) x (nparticle-1) x  (del_qx, del_qy, del_px, del_py, tau )

        return paired_data

    def delta_qorp(self, qorp_list, nparticle, DIM):

        qorp_len = qorp_list.shape[0]
        qorp0 = torch.unsqueeze(qorp_list, dim=0)
        qorpm = torch.repeat_interleave(qorp0, qorp_len, dim=0)

        qorpt = qorpm.permute(1,0,2)

        dqorp = qorpt - qorpm

        dqorp_reduced_index = paired_distance_reduce.get_indices(dqorp.shape)
        dqorp_flatten = paired_distance_reduce.reduce(dqorp, dqorp_reduced_index)

        dqorp_flatten = dqorp_flatten.reshape((nparticle, nparticle - 1, DIM))

        return dqorp_flatten

