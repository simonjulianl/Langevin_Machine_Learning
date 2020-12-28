#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 12:18:24 2020

@author: simon
"""

import numpy as np
from abc import ABC
import sympy as sym

# =================================================
# put common functions here in base class to
# avoid code duplication
class Interaction(ABC):
    '''Interaction base class between particles
    
    Common Parameters
    -----------------
    q_state : np.array
            current state of the position of N X DIM array
    p_state : np.array
        current state of the momentum of N X DIM array
    '''
    def __init__(self, expression):
        self._expression = expression

        
    def energy(self, phase_space,pb):

        term = 0 # sum of separable term
        q_state = phase_space.get_q()
        p_state = phase_space.get_p()

        assert len(q_state.shape) == 3
        # both q_list and p_list must have the same shape and q_state is N X DIM matrix
        for q,p in zip(q_state,p_state) : 
            term += np.sum(eval(self._expression)) # sum across all dimensions

        return term
    
    def evaluate_derivative_q(self, phase_space, pb):
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

        dHdq = np.array([]) #derivative of separable term in N x particle X DIM matrix

        for q,p in zip(q_state,p_state):

            if len(dHdq) == 0 :
                dHdq = np.expand_dims(np.zeros(q.shape),axis=0)

            else :
                temp = np.expand_dims(np.zeros(q.shape),axis=0)

                dHdq = np.concatenate((dHdq, temp))

        dHdq = dHdq.reshape(q_state.shape) # should have the same dimension

        return dHdq

    def evaluate_derivative_p(self, phase_space, pb):
        '''
        Function to calculate dHdp
        
        Returns
        -------
        dHdp: np.array 
            dHdp calculated given the terms, of N X DIM 

        '''
        q_state = phase_space.get_q()
        p_state = phase_space.get_p()
        assert q_state.shape == p_state.shape and len(q_state.shape) == 3

        dHdp = np.array([])#derivative of separable term in N X DIM matrix 
        for q,p in zip(q_state,p_state):

            if len(dHdp) == 0 :

                if(eval(self._derivative_p) == 0):
                    dHdp = np.expand_dims(np.zeros(q.shape),axis=0)
                else:
                    dHdp = np.expand_dims(eval(self._derivative_p), axis=0)

            else :
                if(eval(self._derivative_p) == 0):
                    temp = np.expand_dims(np.zeros(q.shape),axis=0)
                else:
                    temp = np.expand_dims(eval(self._derivative_p), axis=0)

                dHdp = np.concatenate((dHdp, temp))
       
        dHdp = dHdp.reshape(p_state.shape)
        
        return dHdp

                    


