#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 11:35:58 2020

@author: simon
"""
import numpy as np 

def position_verlet(**state) : 
    
    '''
    position verlet integrator method 

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
    
    #get all the constants
    q = state['pos']
    m = state['m']
    p = state['vel'] * m
    
    Hamiltonian = state['hamiltonian']
    time_step = state['time_step']
    
    q = q + time_step / 2 * p #dq/dt
    
    p_list_dummy = np.zeros(state['pos'].shape) # to prevent KE from being integrated
    
    p = p + time_step  * ( -Hamiltonian.get_derivative_q(state['pos'], p_list_dummy) ) #dp/dt
 
    state['pos'] = q ; state['vel'] = p/m # update state 
    
    q = q + time_step / 2 * p #dq/dt
    
    state['pos'] = q ; state['vel'] = p/m # update state 
    
    return state 

position_verlet.name = 'position_verlet' # add attribute to the function for marker
