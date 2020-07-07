#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 12:55:49 2020

@author: simon
"""

from .Interaction import Interaction
import numpy as np

class SHO_interactions(Interaction):
    def __init__(self, k : float):
        '''
        Parameters
        ----------
        k : float
            the spring constant k 
        '''
        try: 
            self._k = float(k)
        except :
            raise Exception('spring constant, k error')
        super().__init__('0.5 * {} * q ** 2.0'.format(self._k))
        self._name = 'Simple Harmonic Oscillation Interactions' 
        #since interaction is a function of r or delta q instead of q, we need to modift the data 

    def energy(self, q_state: object, p_state: object, periodicty: object = False) -> object:
        '''
        function to calculate the term directly
        
        Returns
        -------
        term : float 
            Hamiltonian calculated

        '''
        term = 0 
        N, DIM = q_state.shape
        for i in range(N-1) : # loop for every pair of q1,q2
            for j in range(i+1, N) :
                q1 = q_state[i]
                q2 = q_state[j]
                delta_q = q2 - q1 # calculate delta q 
                if periodicty : # PBC only 
                    if np.abs(delta_q > 0.5):
                        delta_q = delta_q - np.copysign(1.0, delta_q)
                #since it is a radial function
                q = np.dot(delta_q,delta_q) ** 0.5
                term += np.sum(eval(self._expression))     
                
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
        N, DIM = q_state.shape
        dHdq = np.zeros(q_state.shape) #derivative of separable term in N X DIM matrix 
        for i in range(N-1) : # loop for every pair of q1,q2
            for j in range(i+1, N) :
                q1,p1 = q_state[i], p_state[i]
                q2,p2 = q_state[j], p_state[j]
                delta_q,p = q2 - q1, p2 - p1 # calculate delta q and delta p
                q = np.dot(delta_q, delta_q) ** 0.5 # this is the r
                if periodicty : # PBC only 
                    if np.abs(delta_q > 0.5):
                        delta_q = delta_q - np.copysign(1.0, delta_q)
                #since dUdr = dUdx x/r
                dHdq[i] -= eval(self._derivative_q) * delta_q / q
                dHdq[j] += eval(self._derivative_q) * delta_q / q            
     
        return dHdq
 
    