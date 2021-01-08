#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 11:34:53 2020

@author: simon
"""
import numpy as np 

def linear_velocity_verlet(Hamiltonian, **state) :

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
    #get all the constants
    q = state['phase_space'].get_q()
    p = state['phase_space'].get_p()

    #Hamiltonian = state['general_hamiltonian'] # for ML
    tau = state['tau']
    pb_q = state['pb_q']
    boxsize = state['BoxSize']

    print('vv',q,p,tau)

    # p_list_dummy = np.zeros(p.shape) # to prevent KE from being integrated
    # state['phase_space'].set_p(p_list_dummy)

    p = p + tau / 2 * ( -Hamiltonian.dHdq(state['phase_space'], pb_q) ) #dp/dt
    print('vv p 1 update',q,p,tau)

    q = q + tau * p # dq/dt = dK/dp = p

    pb_q.adjust_real(q, boxsize)
    state['phase_space'].set_q(q)

    pb_q.debug_pbc(q, boxsize)

    print('vv q update',q,p,tau)

    p = p + tau / 2 * ( -Hamiltonian.dHdq(state['phase_space'], pb_q) ) #dp/dt

    state['phase_space'].set_q(q) ; state['phase_space'].set_p(p) # update state after 1 step
    print('vv p 2 update',q,p,tau)
    return state 

linear_velocity_verlet.name = 'linear_velocity_verlet' # add attribute to the function for marker
