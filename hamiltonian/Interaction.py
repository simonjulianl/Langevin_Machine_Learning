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
        '''
        base interaction class that will call the expression 

        Parameters
        ----------
        expression : string
            the expression of the term for variable q and p 
            which represents position and momentum respectively 
            otherwise, eval would return error
        '''
        self._expression = expression
        
        def function_wrapper(q, p) :
            ''' helper function for sympy differentiation'''
            return eval(self._expression)
        
        q, p = sym.symbols('q p', real = True) # automatically calculate derivative for q and p expression
        try : 
            self._derivative_q = str(sym.diff(function_wrapper(q, p),q)) # d term dq
            self._derivative_p = str(sym.diff(function_wrapper(q, p),p)) # d term dp
        except :
            raise Exception('Differentiation fail')
        
    def energy(self, q_state, p_state, periodicity = False):
        '''
        function to calculate the term directly
        
        Returns
        -------
        term : float 
            Hamiltonian calculated

        '''
        term = 0 # sum of separable term 
        assert q_state.shape == p_state.shape and len(q_state.shape) == 2
        # both q_list and p_list must have the same shape and q_state is N X DIM matrix
        for q,p in zip(q_state,p_state) : 
            term += np.sum(eval(self._expression)) # sum across all dimensions

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
        dHdq = np.array([]) #derivative of separable term in N X DIM matrix 
        for q,p in zip(q_state,p_state):
            if len(dHdq) == 0 : 
                dHdq = np.expand_dims(eval(self._derivative_q), 0)
            else : 
                temp = np.expand_dims(eval(self._derivative_q), 0)
                dHdq = np.concatenate((dHdq, temp))
       
        dHdq = dHdq.reshape(q_state.shape) # should have the same dimension
        
        return dHdq
    
    def evaluate_derivative_p(self, q_state, p_state, periodicty = False):
        '''
        Function to calculate dHdp
        
        Returns
        -------
        dHdp: np.array 
            dHdp calculated given the terms, of N X DIM 

        '''
        assert q_state.shape == p_state.shape and len(q_state.shape) == 2
        dHdp = np.array([])#derivative of separable term in N X DIM matrix 
        for q,p in zip(q_state,p_state):
            if len(dHdp) == 0 : 
                dHdp = np.expand_dims(eval(self._derivative_p), 0)
            else : 
                temp = np.expand_dims(eval(self._derivative_p), 0)
                dHdp = np.concatenate((dHdp, temp))
       
        dHdp = dHdp.reshape(p_state.shape)
        
        return dHdp

                    


