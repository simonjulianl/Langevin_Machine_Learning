#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 17:21:03 2020

@author: simon
"""
import numpy as np
from .Interaction import Interaction

class Lennard_Jones(Interaction):
    def __init__(self, epsilon : float, sigma : float):
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
            self._cutoff_r = 2.5 * self._sigma
        except :
            raise Exception('sigma / epsilon rror')
        super().__init__('4 * {0} * (({1}/ q) ** 12.0 * ({1}/q) ** 6.0)'.format(self._epsilon, self._sigma))
        self._name = 'Lennard Jones Potential' 
        #since interaction is a function of r or delta q instead of q, we need to modift the data 

    def energy(self, q_state, p_state, periodicty = False):
        '''
        function to calculate the term directly for truncated lennard jones
        
        Returns
        -------
        term : float 
            Hamiltonian calculated

        '''
        truncated_potential = 4 * self._epsilon * ((1/2.5) ** 12.0 - (1/2.5) ** 6.0) 
        term = 0
        N, DIM  = q_state.shape
        for i in range(N-1) : # loop for every pair of q1,q2
            for j in range(i+1, N) :
                q1 = q_state[i]
                q2 = q_state[j]
                delta_q = q2 - q1 # calculate delta q 
                if periodicty : # PBC only 
                    if np.abs(delta_q > 0.5):
                        delta_q = delta_q - np.copysign(1.0, delta_q)
                q = np.dot(delta_q, delta_q) ** 0.5
                term += eval(self._expression) - truncated_potential
                
        return term 
    
    def evaluate_derivative_q(self, q_state, p_state, periodicty = False):
        '''
        Function to calculate dHdq
        
        Returns
        -------
        dHdq: np.array 
            dHdq calculated given the terms of N X DIM 

        '''
        assert q_state.shape == p_state.shape and len(q_state.shape) == 2
        dHdq = np.zeros(q_state.shape) #derivative of separable term in N X DIM matrix 
        N, DIM  = q_state.shape
        for i in range(N-1) : # loop for every pair of q1,q2
            for j in range(i+1, N) :
                q1,p1 = q_state[i], p_state[i]
                q2,p2 = q_state[j], p_state[j]
                delta_q,p = q2 - q1, p2 - p1 # calculate delta q and delta p
                if periodicty : # PBC only 
                    if np.abs(delta_q > 0.5):
                        delta_q = delta_q - np.copysign(1.0, delta_q)
                #since dUdr = dUdx x/r 
                q = np.dot(delta_q,delta_q) ** 0.5
                if q < self._cutoff_r : 
                    dHdq[i] -= eval(self._derivative_q) * delta_q / q
                    dHdq[j] += eval(self._derivative_q) * delta_q / q             
             
        return dHdq
    
