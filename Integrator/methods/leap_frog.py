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
    q = state['phase_space'].get_q()
    p = state['phase_space'].get_p()
    
    Hamiltonian = state['hamiltonian']
    time_step = state['time_step']
    periodicity = state['periodicity']
    
    if periodicity :    
        BoxSize = state['BoxSize']
    else : 
        BoxSize = 1
        
    p_list_dummy = np.zeros(p.shape) # to prevent KE from being integrated
    
    q = q + time_step * p #dq/dt
        
    p = p + time_step  * ( -Hamiltonian.dHdq(q, p_list_dummy, BoxSize, periodicity) ) #dp/dt
    
    state['phase_space'].set_q(q) ; state['phase_space'].set_p(p) # update state 
    
    return state 

leap_frog.name = 'leapfrog'

