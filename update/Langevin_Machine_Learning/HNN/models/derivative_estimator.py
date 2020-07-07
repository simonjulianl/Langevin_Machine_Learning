#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 11:10:59 2020

@author: simon
"""

from torch.autograd import grad
import torch 

def derivative_ML(q_list, p_list, H_potential, H_kinetic) -> tuple: 
    '''
    Calculate the derivative given the function and input
    also acting as the interface for integrator

    Parameters
    ----------
    q_list : torch.tensor
        tensor of position list of N X DIM 
    p_list : torch.tensor
        tensor of momentum list of N x DIM
           

    H_potential : function
        model to approximate H_potential
    H_kinetic : function
        model to approximate H_kinetic

    Returns
    -------
    dqdt_predicted, dpdt_predicted : tuple
        predicted dqdt, dpdt from differentiation

    '''

    #for each function U and KE estimator, we need a set of coordinates 
    coordinates = torch.cat((q_list.unsqueeze(1), p_list.unsqueeze(1)), dim = 1)
    #coordinates of N X 2 for DIM  = 1
    
    U = H_potential(coordinates) # predict the potential
    dpdt_predicted = -grad(U.sum(), q_list, create_graph = True)[0] # dpdt = -dH/dq
    #need to retain the graph to calculcate the in-graph gradient
    
    KE = H_kinetic(coordinates) # only potential part of the hamiltonian is used here 
    dqdt_predicted = grad(KE.sum(), p_list, create_graph = True)[0] # dqdt = -dH/dp

    return (dqdt_predicted, dpdt_predicted) # all data is arrange in q p manner