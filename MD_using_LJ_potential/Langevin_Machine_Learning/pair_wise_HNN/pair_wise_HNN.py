#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import numpy as np


class pair_wise_HNN:

    def __init__(self,hamiltonian,network):
        '''
        Hamiltonian class for all potential and kinetic interactions
        '''
        self.network = network
        self.noML_hamiltonian = hamiltonian
        self.tau = 0


    # def set_tau(self,t):
    #     self.tau = t

    def train(self):
        self.network.train() # pytorch network for training

    def eval(self):
        self.network.eval()

    # def total_energy(self, phase_space, pb):
    #     '''
    #     get the hamiltonian which is define as H(p,q) for every separable terms
    #
    #     Returns
    #     -------
    #     H : float
    #         H is the hamiltonian of the states with separable terms
    #     '''
    #     return self.noML_hamiltonian.total_energy(phase_space,pb)

    def dHdq(self, phase_space, pb):

        noML_force = self.noML_hamiltonian.dHdq(phase_space,pb)
        print('no ML',noML_force)

        #noML_force = torch.from_numpy(noMLdHdq)

        data = self.phase_space2data( phase_space, pb)
        predict = self.network(data)

        corrected_force = noML_force + predict

        # print('noML',noMLdHdq)
        # # print('ML',self.MLdHdq)
        # print('noML+ML',noMLdHdq + self.MLdHdq.detach().cpu().numpy())
        #return noMLdHdq + self.MLdHdq.detach().cpu().numpy()
        return corrected_force


    def phase_space2data(self, phase_space, pb):

        print('phase_space2data')
        print(phase_space.get_q(), phase_space.get_p())

        init_q = self.phase_space.set_q(self.q_list)
        init_p = self.phase_space.set_p(self.p_list)
        print(init_q)
        N, N_particle, DIM = init_q.shape

        delta_init_q = np.zeros( (N, N_particle , (N_particle - 1), DIM) )
        delta_init_p = np.zeros( (N, N_particle , (N_particle - 1), DIM) )

        for z in range(N):

            delta_init_q_, _ = self._setting['pb_q'].paired_distance_reduced(init_q[z]/self._setting['BoxSize']) #reduce distance
            delta_init_q_ = delta_init_q_ * self._setting['BoxSize']
            delta_init_p_, _ = self._setting['pb_q'].paired_distance_reduced(init_p[z]/self._setting['BoxSize']) #reduce distance
            delta_init_p_ = delta_init_p_ * self._setting['BoxSize']

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
                        delta_init_q[z][i][x] = delta_init_q_[i,j,:]
                        delta_init_p[z][i][x] = delta_init_p_[i,j,:]

                        x=x+1

        # print('delta_init')
        # print(delta_init_q)
        # print(delta_init_p)

        # tau : #this is big time step to be trained
        # to add paired data array
        tau = np.array([self._setting['tau'] * self._setting['iterations']] * N_particle * (N_particle - 1))
        tau = tau.reshape(-1, N_particle, (N_particle - 1), 1) # broadcasting
        print('tau',tau.shape)
        # print('concat')
        paired_data_ = np.concatenate((delta_init_q,delta_init_p),axis=-1) # N (nsamples) x N_particle x (N_particle-1) x (del_qx, del_qy, del_px, del_py)
        # print(paired_data_)
        # print(paired_data_.shape)
        paired_data = np.concatenate((paired_data_,tau),axis=-1) # nsamples x N_particle x (N_particle-1) x  (del_qx, del_qy, del_px, del_py, tau )
        paired_data = paired_data.reshape(-1,paired_data.shape[3]) # (nsamples x N_particle) x (N_particle-1) x  (del_qx, del_qy, del_px, del_py, tau )

        print('=== input data for ML : del_qx del_qy del_px del_py tau ===')
        print(paired_data)
        print(paired_data.shape)

        return paired_data


    # def d2Hdq2(self, phase_space, pb):
    #     '''
    #     Function to get d2Hdq2 for every separable terms
    #
    #     Returns
    #     -------
    #     d2Hdq2 : float
    #         d2Hdq2 is the second derivative of H with respect to q for N x N_particle x DIM dimension
    #     '''
    #     noMLd2Hdq2 = self.noML_hamiltonian(phase_space,pb)
    #
    #     MLd2Hdq2 = self.predict_residue_d2Hdq2(tau,phase_space,pb)
    #
    #     return noMLd2Hdq2+MLd2Hdq2


