#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 12:18:24 2020

@author: simon
"""

"""
defunct class, do not use
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
        # HK: disable the class
        print('class is defunct, do not use, exiting')
        quit()
        
        #print('Interaction.py expression',expression)
        self._expression = expression

    """
    # HK begin comment out code
    def energy(self, phase_space,pb):
        '''
        function to calculate the term directly
        
        Returns
        -------
        term : float 
            Hamiltonian calculated

        '''
        term = 0 # sum of separable term
        # both q_list and p_list must have the same shape and q_state is N X particle X DIM matrix
        #print('interation.py phase_space',phase_space)
        q_state = phase_space.get_q()
        p_state = phase_space.get_p()
        #print('interation.py q_state',q_state.shape)
        #print('interation.py p_state',p_state.shape)
        assert q_state.shape == p_state.shape and len(q_state.shape) == 3
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
        #print('interation.py evaluate_derivative_q',q_state.shape)
        dHdq = np.array([]) #derivative of separable term in N x particle X DIM matrix
        #dHdq = np.zeros(q_state.shape)
        #dHdq = np.array([[],[]])

        for q,p in zip(q_state,p_state):
            #print('interation.py dHdq', q.shape, p.shape,dHdq.shape)
            if len(dHdq) == 0 :
                #print('interation.py self._derivative_q', self._derivative_q)

                dHdq = np.expand_dims(np.zeros(q.shape),axis=0)
                #print('interation.py len(dHdq)=0 if ', dHdq)

                    #print('interation.py len(dHdq)=0 else', dHdq)
            else :
                temp = np.expand_dims(np.zeros(q.shape),axis=0)
                    #print('interation.py temp len(dHdq) if ', temp)

                dHdq = np.concatenate((dHdq, temp))
                #print('interation.py len(dHdq)', dHdq)

        dHdq = dHdq.reshape(q_state.shape) # should have the same dimension
        #print('interation.py evaluate_derivative_q dHdq',dHdq.shape)

        return dHdq

    def evaluate_second_derivative_q(self, phase_space, pb):
        '''
        Function to calculate dHdp
        
        Returns
        -------
        dHdp: np.array 
            dHdp calculated given the terms, of N X DIM 

        '''
        q_state = phase_space.get_q()
        #print(q_state.shape)
        p_state = phase_space.get_p()
        assert q_state.shape == p_state.shape and len(q_state.shape) == 3
        #print('interation.py evaluate_derivative_q',q_state.shape)
        d2Hdq2 = np.array([]) #derivative of separable term in N x particle X DIM matrix
        #dHdq = np.zeros(q_state.shape)
        #dHdq = np.array([[],[]])

        for q,p in zip(q_state,p_state):
            #print('interation.py dHdq', q.shape, p.shape,dHdq.shape)
            if len(d2Hdq2) == 0 :
                #print('interation.py self._derivative_q', self._derivative_q)

                d2Hdq2 = np.expand_dims(np.zeros((2*q.shape[0],2*q.shape[0])),axis=0)
                #print('interation.py len(dHdq)=0 if ', dHdq)

                    #print('interation.py len(dHdq)=0 else', dHdq)
            else :
                temp = np.expand_dims(np.zeros((2*q.shape[0],2*q.shape[0])),axis=0)
                    #print('interation.py temp len(dHdq) if ', temp)

                d2Hdq2 = np.concatenate((d2Hdq2, temp))
                #print('interation.py len(dHdq)', dHdq)

        d2Hdq2 = d2Hdq2.reshape(np.zeros((q_state.shape[0],2*q_state.shape[1],2*q_state.shape[1])).shape) # should have the same dimension
        #print('interation.py evaluate_derivative_q dHdq',dHdq.shape)
        
        return d2Hdq2
        
        # HK : end comment out code
        """           


