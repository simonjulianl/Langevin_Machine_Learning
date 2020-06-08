#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 11:37:12 2020

@author: simon
"""
import numpy as np 

def leap_frog(**state) :
    
    '''
    leap frog integration method, assuming v1/2 and q0 can be obtained 

    Parameters
    ----------
    **state : dict
        all the configuration describing the state
        require : 
            -Hamiltonian : hamiltonian 
                list of functions modelling the energy
            -m : float 
                mass
            -q : np.array of N X DIM 
            -p : np.array of N X DIM 

    Returns
    -------
    state : dict
        updated configuration state after 1 integration method 

    '''
    q = state['pos']
    m = state['m']
    p = state['vel'] * m
    
    Hamiltonian = state['hamiltonian']
    time_step = state['time_step']
    
    p_list_dummy = np.zeros(state['pos'].shape) # to prevent KE from being integrated
    
    q = q + time_step * p #dq/dt
    
    state['pos'] = q ; state['vel'] = p/m # update state 
    
    p = p + time_step  * ( -Hamiltonian.get_derivative_q(state['pos'], p_list_dummy) ) #dp/dt
    
    state['pos'] = q ; state['vel'] = p/m # update state 
    
    return state 

leap_frog.name = 'leapfrog'

