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

def velocity_verlet_ML(**state) : 
    '''
    velocity verlet integration method using ML approximation 

    Parameters
    ----------
    **state : dict
        all the configuration describing the state
        require : 
            -m : float 
                mass
            -q : np.array of N X DIM 
            -p : np.array of N X DIM 
    
    ML_integrator : ML_integrator function 
    * this is taken to be fixed from the available model
            
    Returns
    -------
    state : dict
        updated configuration state after 1 integration method 

    '''
    
    device = state.get('device', 'cpu')
    q = torch.tensor(state['pos'], dtype = torch.float32).requires_grad_(True).to(device)
    m = state['m']
    p = torch.tensor(state['vel'], dtype = torch.float32).requires_grad_(True).to(device) * m
    
    from ..HNN.models.MLP2H_Separable_Hamil_VV import MLP2H_Separable_Hamil_VV
    
    import os 
    currpath = os.getcwd() # the ML paths to be saved with the same path as methods
    best_setting = torch.load(currpath + '/states/VV_ML_best.pth')
    #actually this is not efficient for loading over and over
    
    ML_integrator = MLP2H_Separable_Hamil_VV(2, 20)
    ML_integrator.load_state_dict(best_setting['state_dict'])
    ML_integrator.eval()
    q_next, p_next = ML_integrator(q, p)
    state['pos'] = q_next.cpu().detach().numpy()
    state['vel'] = p_next.cpu().detach().numpy() / m
    
    return state 

velocity_verlet_ML.name = 'velocity_verlet_ML'

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

def position_verlet_ML(**state) : 
    '''
    position verlet integration method using ML approximation 

    Parameters
    ----------
    **state : dict
        all the configuration describing the state
        require : 
            -m : float 
                mass
            -q : np.array of N X DIM 
            -p : np.array of N X DIM 
    
    ML_integrator : ML_integrator function 
    * this is taken to be fixed from the available model
            
    Returns
    -------
    state : dict
        updated configuration state after 1 integration method 

    '''
    
    device = state.get('device', 'cpu')
    q = torch.tensor(state['pos'], dtype = torch.float32).requires_grad_(True).to(device)
    m = state['m']
    p = torch.tensor(state['vel'], dtype = torch.float32).requires_grad_(True).to(device) * m
    
    from ..HNN.models.MLP2H_Separable_Hamil_PV import MLP2H_Separable_Hamil_PV
    
    import os 
    currpath = os.getcwd() # the ML paths to be saved with the same path as methods
    best_setting = torch.load(currpath + '/states/PV_ML_best.pth')
    #actually this is not efficient for loading over and over
    
    ML_integrator = MLP2H_Separable_Hamil_PV(2, 20)
    ML_integrator.load_state_dict(best_setting['state_dict'])
    ML_integrator.eval()
    q_next, p_next = ML_integrator(q, p)
    state['pos'] = q_next.cpu().detach().numpy()
    state['vel'] = p_next.cpu().detach().numpy() / m
    
    return state 

position_verlet_ML.name = 'velocity_verlet_ML'

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
    q = torch.tensor(state['pos'], dtype = torch.float32).requires_grad_(True).to(device)
    m = state['m']
    p = torch.tensor(state['vel'], dtype = torch.float32).requires_grad_(True).to(device) * m
    
    from ..HNN.models.MLP2H_Separable_Hamil_LF import MLP2H_Separable_Hamil_LF
    
    import os 
    currpath = os.getcwd() # the ML paths to be saved with the same path as methods
    best_setting = torch.load(currpath + '/states/LF_ML_best.pth')
    
    ML_integrator = MLP2H_Separable_Hamil_LF(2, 20)
    ML_integrator.load_state_dict(best_setting['state_dict'])
    ML_integrator.eval()
    q_next, p_next = ML_integrator(q, p)
    state['pos'] = q_next.cpu().detach().numpy()
    state['vel'] = p_next.cpu().detach().numpy() / m
    
    return state 

leap_frog_ML.name = 'leapfrog_ML'
