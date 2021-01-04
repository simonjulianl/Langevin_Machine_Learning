#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

class pair_wise_HNN:

    def __init__(self, noML_hamiltonian, MLdHdq):
        '''
        Hamiltonian class for all potential and kinetic interactions
        '''
        self.noML_hamiltonian = noML_hamiltonian  # for every separable terms possible
        self.MLdHdq = MLdHdq
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
        # print('noML',noMLdHdq)
        # # print('ML',self.MLdHdq)
        # print('noML+ML',noMLdHdq + self.MLdHdq.detach().cpu().numpy())
        return noMLdHdq + self.MLdHdq.detach().cpu().numpy()

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

    # def predict_residue_dHdq(self,tau,phase_space,pb):
    #
    #     # pass data though MLP to predict dHdq
    #     predict_dHdq = self.pair_wise_MLP(i)
    #
    #     return predict_dHdq

