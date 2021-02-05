#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch

class hamiltonian:
    '''Class container for Hamiltonian

    Common Parameters
    -----------------
    q_list : np.array
            q_list np.array of N (nsamples) x N_particle x DIM matrix which is the position of the states
    p_list : np.array
        p_list np.array of N (nsamples) x N_particle x DIM matrix which is the momentum of the states
    '''
    _obj_count = 0

    def __init__(self):
    # def __init__(self, phase_p):
        '''
        Hamiltonian class for all potential and kinetic interactions
        '''

        # print('call hamiltonian obj')

        hamiltonian._obj_count += 1
        assert (hamiltonian._obj_count == 1),type(self).__name__ + " has more than one object"

        self.hamiltonian_terms = []  # for every separable terms possible
        # self.phase_space = phase_p (reference)
        # self.phase_pace = phase_space() (copy)

    def hi(self):
        print('hi')
        return 1

    def get_terms(self):
        return self.hamiltonian_terms

    def append(self, term):
        '''
        helper function to add into the hamiltonian terms
        '''
        # print('call hamiltonian append')
        self.hamiltonian_terms.append(term)

    def total_energy(self, phase_space):
        '''
        get the hamiltonian which is define as H(p,q) for every separable terms

        Returns
        -------
        H : float
            H is the hamiltonian of the states with separable terms
        '''
        H = 0
        # print('call hamiltonian def energy')
        for term in self.hamiltonian_terms:
            # print('term ', term)
            H += term.energy(phase_space)
            # print('H ', H)
        return H

    def dHdq(self, phase_space):
        '''
        Function to get dHdq for every separable terms

        Returns
        -------
        dHdq : float
            dHdq is the derivative of H with respect to q for N x N_particle x DIM dimension
        '''
        # print('call hamiltonian def dHdq')
        q_list = phase_space.get_q()
        dHdq = torch.zeros(q_list.shape) #- need same type as q_list

        for term in self.hamiltonian_terms:
            # print('hamiltonian dHdq', term)
            dHdq += term.evaluate_derivative_q(phase_space)
            # print('hamiltonian dHdq', dHdq)
        return dHdq

    def d2Hdq2(self, phase_space):
        '''
        Function to get d2Hdq2 for every separable terms

        Returns
        -------
        d2Hdq2 : float
            d2Hdq2 is the second derivative of H with respect to q for N x N_particle x DIM dimension
        '''
        q_list = phase_space.get_q()

        nsamples, nparticle, DIM = q_list.shape
        d2Hdq2 = torch.zeros((nsamples, DIM * nparticle, DIM * nparticle))

        for term in self.hamiltonian_terms:
            # print('hamiltonian', term)
            d2Hdq2 += term.evaluate_second_derivative_q(phase_space)
            # print('hamiltonian', d2Hdq2)

        return d2Hdq2

    def __repr__(self):
        ''' return list of terms'''
        terms = ''
        for term in self.hamiltonian_terms:
            terms += term._name + '\n'
        return terms
