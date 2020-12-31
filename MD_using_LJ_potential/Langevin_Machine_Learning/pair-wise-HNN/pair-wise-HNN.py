#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class pair_wise_HNN:
    '''Class container for Hamiltonian

    Common Parameters
    -----------------
    q_list : np.array
            q_list np.array of N (nsamples) x N_particle x DIM matrix which is the position of the states
    p_list : np.array
        p_list np.array of N (nsamples) x N_particle x DIM matrix which is the momentum of the states
    '''

    def __init__(self,noML_hamiltonian,pair_wise_MLP):
        '''
        Hamiltonian class for all potential and kinetic interactions
        '''
        self.noML_hamiltonian = noML_hamiltonian  # for every separable terms possible
        self.pair_wise_MLP = pair_wise_MLP
        self.tau = 0

    def set_tau(self,t):
        self.tau = t

    def total_energy(self, phase_space, pb):
        '''
        get the hamiltonian which is define as H(p,q) for every separable terms

        Returns
        -------
        H : float
            H is the hamiltonian of the states with separable terms
        '''
        return self.noML_hamiltonian.total_energy(phase_space,pb)

    def dHdq(self, phase_space, pb):
        '''
        Function to get dHdq for every separable terms

        Returns
        -------
        dHdq : float
            dHdq is the derivative of H with respect to q for N x N_particle x DIM dimension
        '''

        noMLdHdq = self.noML_hamiltonian.dHdq(phase_space,pb)
        MLdHdq = self.predict_residue_dHdq(self.tau,phase_space,pb)

        return noMLdHdq + MLdHdq

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

    def predict_residue_dHdq(self,tau,phase_space,pb):

        # generate data by pairing up all particles - get delta_q
        q_state = phase_space.get_q()
        p_state = phase_space.get_p()
        N, N_particle, DIM = q_state.shape

        # pass data though MLP to predict dHdq
        predict_dHdq = np.zeros(q_state.shape)

        for z in range(N):

            delta_q, _ = pb.paired_distance_reduced(q_state[z])
            delta_p, _ = pb.paired_distance_reduced(p_state[z])

            paired_data = np.array([delta_q,delta_p,tau])

            for i in paired_data:
                predict_dHdq[z][i] = self.pair_wise_MLP(i)

        return predict_dHdq

    def __repr__(self):
        ''' return list of terms'''
        terms = ''
        for term in self.hamiltonian_terms:
            terms += term._name + '\n'
        return terms