#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 11:34:53 2020

@author: simon
"""
import numpy as np 

def velocity_verlet(**state) :
    
    '''
    velocity verlet integrator method 

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
    #print("vv")
    #print(state)
    #get all the constants
    q = state['phase_space'].get_q()
    p = state['phase_space'].get_p()
    Hamiltonian = state['hamiltonian']
    time_step = state['time_step']
    pb_q = state['pb_q']
    boxsize = state['BoxSize']

    #print('velocity_verlet.py input q', q)
    #print('velocity_verlet.py input p', p)

    p_list_dummy = np.zeros(p.shape) # to prevent KE from being integrated
    state['phase_space'].set_p(p_list_dummy)

    #print('velocity_verlet.py before update p', p)
    p = p + time_step / 2 * ( -Hamiltonian.dHdq(state['phase_space'], state['pb_q']) ) #dp/dt
    #print('velocity_verlet.py update p', p)

    #print('velocity_verlet.py before update q', q)
    q = q + time_step * p # dq/dt = dK/dp = p
    #print('velocity_verlet.py update q', q)

    pb_q.adjust_real(q, boxsize)
    #print('velocity_verlet.py after pbc q', q)
    state['phase_space'].set_q(q)

    pb_q.debug_pbc(q, boxsize)

    #print('velocity_verlet.py before update p', p)
    p = p + time_step / 2 * ( -Hamiltonian.dHdq(state['phase_space'], state['pb_q']) ) #dp/dt
    #print('velocity_verlet.py update p', p)

    state['phase_space'].set_q(q) ; state['phase_space'].set_p(p) # update state after 1 step 
    
    return state 

velocity_verlet.name = 'velocity_verlet' # add attribute to the function for marker
