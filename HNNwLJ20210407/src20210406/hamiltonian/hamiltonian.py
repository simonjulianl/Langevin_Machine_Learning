#!/usr/bin/env python3

import torch

class hamiltonian:

    _obj_count = 0

    def __init__(self):
        '''
        Hamiltonian class for all potential and kinetic interactions
        '''

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
        append function to add into the hamiltonian terms
        '''

        self.hamiltonian_terms.append(term)

    def total_energy(self, phase_space):
        '''
        total_energy function to get the hamiltonian which is define as H(p,q) for every separable terms

        Returns
        -------
        H : float
            H is the hamiltonian of the states with separable terms
            shape is [nsamples]
        '''

        H = 0

        for term in self.hamiltonian_terms:

            H += term.energy(phase_space)

        return H

    def dHdq(self, phase_space):
        '''
        dHdq function to get dHdq for every separable terms

        Returns
        -------
        dHdq : float
            dHdq is the derivative of H with respect to q
            shape is [nsamples, nparticle, DIM]
        '''

        q_list = phase_space.get_q()
        dHdq = torch.zeros(q_list.shape) #- need same type as q_list
        # dHdq = 0

        for term in self.hamiltonian_terms:

            dHdq += term.evaluate_derivative_q(phase_space)

        return dHdq

    def d2Hdq2(self, phase_space):
        '''
        Function to get d2Hdq2 for every separable terms

        Returns
        -------
        d2Hdq2 : float
            d2Hdq2 is the second derivative of H with respect to q
            shape is [nsamples, nparticle, DIM]
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
