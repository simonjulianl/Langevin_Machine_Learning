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
    q = state['phase_space'].get_q()
    p = state['phase_space'].get_p()

    Hamiltonian = state['hamiltonian']
    time_step = state['time_step']
    periodicity = state['periodicity']
    
    if periodicity :    
        BoxSize = state['BoxSize']
    else : 
        BoxSize = 1
        
    q = q + time_step / 2 * p #dq/dt
    
    p_list_dummy = np.zeros(p.shape) # to prevent KE from being integrated
    
    p = p + time_step  * ( -Hamiltonian.dHdq(q, p_list_dummy, BoxSize, periodicity) ) #dp/dt
     
    q = q + time_step / 2 * p #dq/dt
    
    state['phase_space'].set_q(q) ; state['phase_space'].set_p(p) # update state 
    
    return state 

position_verlet.name = 'position_verlet' # add attribute to the function for marker

