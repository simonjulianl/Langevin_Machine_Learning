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
    print('position_verlet.py q,p',q.shape,p.shape)
    Hamiltonian = state['hamiltonian']
    time_step = state['time_step']
    periodicity = state['periodicity']

    BoxSize = state['BoxSize']

    q = q + time_step / 2 * p #dq/dt
    print('position_verlet.py q', q.shape)

    p_list_dummy = np.zeros(p.shape) # to prevent KE from being integrated
    state['phase_space'].set_p(p_list_dummy)
    print('position_verlet.py p_list_dummy', p_list_dummy.shape)

    print('position_verlet.py dHdq',-Hamiltonian.dHdq(state['phase_space'], BoxSize, periodicity))
    p = p + time_step  * (-Hamiltonian.dHdq(state['phase_space'], BoxSize, periodicity)  ) #dp/dt
    print('position_verlet.py p', p.shape)

    q = q + time_step / 2 * p #dq/dt
    print('position_verlet.py q', q.shape)

    state['phase_space'].set_q(q) ; state['phase_space'].set_p(p) # update state

    print('position_verlet.py state',state)
    return state

position_verlet.name = 'position_verlet' # add attribute to the function for marker

