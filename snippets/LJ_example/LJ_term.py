#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 17:21:03 2020

@author: simon
"""
import numpy as np
from Interaction import Interaction

class LJ_term(Interaction):
    def __init__(self, epsilon : float, sigma : float, exponent : float, boxsize : float):
        '''
        Parameters
        ----------
        epsilon : float
            depth of potential well
        sigma : float
            finite distance at which the inter-particle potential is zero
        '''
        try:
            self._epsilon  = float(epsilon)
            self._sigma    = float(sigma)
            self._boxsize  = float(boxsize)
            self._exponent = float(exponent)

        except :
            raise Exception('sigma / epsilon rror')

        super().__init__('1.0 / q ** {0} '.format(self._exponent))
        print('Lennard_Jones.py call LJ potential')
        self._name = 'Lennard Jones Potential'
        #since interaction is a function of r or delta q instead of q, we need to modift the data


    def energy(self, phase_space, pb):
        '''
        function to calculate the term directly for truncated lennard jones
        
        Returns
        -------
        term : float 
            Hamiltonian calculated

        '''
        xi_state = phase_space.get_q()
        term = np.zeros(xi_state.shape[0])
        # N : number of data in batch
        # n_particle : number of particles
        # DIM : dimension of xi
        N,n_particle,DIM  = xi_state.shape # ADD particle
        for k in range(N):
            pb.adjust(xi_state[k])
            _, q = pb.paired_distance(xi_state[k])
            term[k] += np.nansum(eval(self._expression))*0.5

        term = term * (4*self._epsilon)*((self._sigma/self._boxsize)**self._exponent )
        return term

    def evaluate_derivative_q(self, phase_space, pb):
        '''
        Function to calculate dHdq
        
        Returns
        -------
        dphidxi: np.array 
            dphidxi calculated given the terms of N X DIM 

        '''
        xi_state = phase_space.get_q()
        p_state  = phase_space.get_p()
        dphidxi = np.zeros(xi_state.shape) #derivative of separable term in N X DIM matrix
        N, particle,DIM  = xi_state.shape
        print('Lennard_Jones.py evaluate_derivative_q xi_state',xi_state)

        for k in range(N):
            pb.adjust(xi_state[k])
            delta_xi, xi = pb.paired_distance(xi_state[k])
            #dphidxi[k] = np.nansum(eval(self._derivative_xi),axis=1) * delta_q / np.sum(q,axis=1)

        print('Lennard_Jones.py evaluate_derivative_q dHdq end', dHdq.shape)
        #print('Lennard_Jones.py evaluate_derivative_q dHdq end', dHdq)
        return dHdq

