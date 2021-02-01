#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 11:34:53 2020

@author: simon
"""
import numpy as np 

def pair_wise_linear_velocity_verlet(**state) :
    
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
    # q = state['phase_space'].get_q()
    # p = state['phase_space'].get_p()
    q, p = state['init_config']
    state['phase_space'].set_q(q)
    state['phase_space'].set_p(p)

    Hamiltonian = state['HNN']
    tau = state['tau'] * state['iterations'] #this is big time step to be trained
    print('pair wise vv tau',tau)

    pb_q = state['pb_q']
    boxsize = state['BoxSize']

    p_list_dummy = np.zeros(p.shape) # to prevent KE from being integrated
    state['phase_space'].set_p(p_list_dummy)

    p = p + tau / 2 * ( -Hamiltonian.dHdq(state['phase_space'], state['pb_q']) ) #dp/dt
    #print('update p',q,p)

    q = q + tau * p # dq/dt = dK/dp = p

    pb_q.adjust_real(q, boxsize)
    state['phase_space'].set_q(q)

    pb_q.debug_pbc(q, boxsize)
    #print('update q',q,p)

    p = p + tau / 2 * ( -Hamiltonian.dHdq(state['phase_space'], state['pb_q']) ) #dp/dt
    #print('update p',q,p)

    state['phase_space'].set_q(q) ; state['phase_space'].set_p(p) # update state after 1 step 
    
    return state 

pair_wise_linear_velocity_verlet.name = 'pair_wise_linear_velocity_verlet' # add attribute to the function for marker
