#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class Hamiltonian:
    '''Class container for Hamiltonian

    Common Parameters
    -----------------
    q_list : np.array
            q_list np.array of N (nsamples) x N_particle x DIM matrix which is the position of the states
    p_list : np.array
        p_list np.array of N (nsamples) x N_particle x DIM matrix which is the momentum of the states
    '''

    def __init__(self):
        '''
        Hamiltonian class for all potential and kinetic interactions
        '''
        self.hamiltonian_terms = []  # for every separable terms possible

    def append(self, term):
        '''
        helper function to add into the hamiltonian terms
        '''
        self.hamiltonian_terms.append(term)

    def total_energy(self, phase_space, pb):
        '''
        get the hamiltonian which is define as H(p,q) for every separable terms

        Returns
        -------
        H : float
            H is the hamiltonian of the states with separable terms
        '''
        H = 0

        for term in self.hamiltonian_terms:
            H += term.energy(phase_space, pb)

        return H

    def dHdq(self, phase_space, pb):
        '''
        Function to get dHdq for every separable terms

        Returns
        -------
        dHdq : float
            dHdq is the derivative of H with respect to q for N x N_particle x DIM dimension
        '''
        q_list = phase_space.get_q()
        dHdq = np.zeros(q_list.shape)

        for term in self.hamiltonian_terms:
            # print(term)
            dHdq += term.evaluate_derivative_q(phase_space, pb)
            # print(dHdq)
        return dHdq

    def d2Hdq2(self, phase_space, pb):
        '''
        Function to get d2Hdq2 for every separable terms

        Returns
        -------
        d2Hdq2 : float
            d2Hdq2 is the second derivative of H with respect to q for N x N_particle x DIM dimension
        '''
        q_list = phase_space.get_q()

        N, N_particle, DIM = q_list.shape
        d2Hdq2 = np.zeros((N, DIM * N_particle, DIM * N_particle))

        for term in self.hamiltonian_terms:
            d2Hdq2 += term.evaluate_second_derivative_q(phase_space, pb)

        return d2Hdq2

    def __repr__(self):
        ''' return list of terms'''
        terms = ''
        for term in self.hamiltonian_terms:
            terms += term._name + '\n'
        return terms
