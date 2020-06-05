#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 11:11:49 2020

@author: simon
"""

import numpy as np

'''list of functions for MD Lpq splitting 
or to integrate the Newton's equation of motion
'''

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
    
    #get all the constants
    q = state['pos']
    m = state['m']
    p = state['vel'] * m
    
    Hamiltonian = state['hamiltonian']
    time_step = state['time_step']
    
    p_list_dummy = np.zeros(state['pos'].shape) # to prevent KE from being integrated
    
    p = p + time_step / 2 * ( -Hamiltonian.get_derivative_q(state['pos'], p_list_dummy) ) #dp/dt
 
    q = q + time_step * p #dq/dt
    
    state['pos'] = q ; state['vel'] = p/m # update state 
    
    p = p + time_step / 2 * ( -Hamiltonian.get_derivative_q(state['pos'], p_list_dummy) ) #dp/dt
    
    state['pos'] = q ; state['vel'] = p/m # update state 
    
    return state 

velocity_verlet.name = 'velocity_verlet' # add attribute to the function for marker

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

import torch 

def leap_frog_ML(**state):
    '''
    leap frog ML integration method, assuming v1/2 and q0 can be obtained 

    Parameters
    ----------
    **state : dict
        all the configuration describing the state
        require : 
            -m : float 
                mass
            -q : np.array of N X DIM 
            -p : np.array of N X DIM 
            -ML_integrator : ML_integrator function 
            
    Returns
    -------
    state : dict
        updated configuration state after 1 integration method 

    '''
    
    device = state.get('device', 'cpu')
    q = torch.tensor(state['pos']).to(device)
    m = state['m']
    p = torch.tensor(state['vel']).to(device) * m
    
    try : 
        ML_integrator = state['ML_integrator']
    except :
        raise Exception('ML_integrator not found')
    
    q_next, p_next = ML_integrator(q, p)
    state['pos'] = q_next.cpu().detach().numpy()
    state['vel'] = p_next.cpu().detach().numpy() / m
    
    return state 

leap_frog_ML.name = 'leapfrog_ML'
