#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 17:21:03 2020

@author: simon
"""
import numpy as np
from .Interaction import Interaction

class Lennard_Jones(Interaction):
    def __init__(self, epsilon : float, sigma : float, BoxSize : float):
        '''
        Parameters
        ----------
        epsilon : float
            depth of potential well
        sigma : float
            finite distance at which the inter-particle potential is zero
        '''
        try:
            self._epsilon = float(epsilon)
            self._sigma = float(sigma)
            self._BoxSize = float(BoxSize)

            #self._cutoff_r = 2.5 * self._sigma
        except :
            raise Exception('sigma / epsilon rror')

        super().__init__('4 * {0}  * ( ({1}/ (q * {2})) ** 12.0 - ({1}/(q * {2})) ** 6.0)'.format(self._epsilon, self._sigma,self._BoxSize))
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
        #truncated_potential = 4 * self._epsilon * ((1/2.5) ** 12.0 - (1/2.5) ** 6.0)
        #print('Lennard_Jones.py energy phase_space',phase_space)
        q_state = phase_space.get_q()
        term = np.zeros(q_state.shape[0])
        #print('Lennard_Jones.py energy q_state',q_state.shape)
        N, particle,DIM  = q_state.shape # ADD particle
        #print('Lennard_Jones.py energy', q_state)
        print('Lennard_Jones.py self._expression', self._expression)
        for k in range(N):
            pb.adjust(q_state[k])
            delta_q, q=pb.paired_distance(q_state[k])
            print('Lennard_Jones.py q',q)
            term[k] = np.nansum(eval(self._expression))*0.5
            print('Lennard_Jones.py term',term[k])

        return term

    def evaluate_derivative_q(self, phase_space,pb):
        '''
        Function to calculate dHdq
        
        Returns
        -------
        dHdq: np.array 
            dHdq calculated given the terms of N X DIM 

        '''
        q_state = phase_space.get_q()
        p_state = phase_space.get_p()
        assert q_state.shape == p_state.shape and len(q_state.shape) == 3
        dHdq = np.zeros(q_state.shape) #derivative of separable term in N X DIM matrix
        N, particle,DIM  = q_state.shape
        #print('Lennard_Jones.py evaluate_derivative_q p_state', p_state)
        print('Lennard_Jones.py evaluate_derivative_q q_state',q_state)
        for k in range(N):
            pb.adjust(q_state[k])
            delta_q, q = pb.paired_distance(q_state[k])
            print('evaluate_derivtak 2ative_q q',q)
            print('evaluate_derivative_q delta_q', delta_q)
            print('evaluate_derivative_q eval', self._derivative_q)
            print('evaluate_derivative_q eval', eval(self._derivative_q))
            #print('evaluate_derivative_q eval',eval(self._derivative_q)/self._BoxSize)
            print('evaluate_derivative_q eval',np.nansum(eval(self._derivative_q)/self._BoxSize,axis=1))
            print('evaluate_derivative_q eval delta_q / q',delta_q / np.sum(q,axis=1))
            dHdq[k] = np.nansum(eval(self._derivative_q)/self._BoxSize,axis=1) * delta_q / np.sum(q,axis=1)
            print('evaluate_derivative_q dHdq',dHdq[k])

        print('Lennard_Jones.py evaluate_derivative_q dHdq end', dHdq.shape)
        #print('Lennard_Jones.py evaluate_derivative_q dHdq end', dHdq)
        return dHdq

