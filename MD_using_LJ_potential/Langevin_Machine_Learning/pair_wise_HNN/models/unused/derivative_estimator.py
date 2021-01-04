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
    #print('derivative estimator.py q_list',q_list.shape)
    #print('derivative estimator.py q_list',q_list)
    #print('derivative estimator.py p_list', p_list.shape)
    #print('derivative estimator.py p_list', p_list)
    coordinates = torch.cat((q_list, p_list), dim = -1)
    #print('derivative estimator.py coordinates', coordinates)
    #coordinates of N X 2 for DIM  = 1
    # coordinates of N_particles X DIM X 2 for DIM  = 2
    
    U = H_potential(coordinates) # predict the potential
    #U.sum(1) : make batch x 1 shape  batch shape
    #U = torch.cat(4*[U],dim=-1)
    #print(q_list)

    # need to retain the graph to calculcate the in-graph gradient
    dpdt_predicted = -grad(U.sum(1), q_list, grad_outputs=torch.ones_like(U.sum(1)), create_graph=True)[0]  # dpdt = -dH/dq
    #print('q dot',dpdt_predicted)

    KE = H_kinetic(coordinates) # only potential part of the hamiltonian is used here

    #print('KE',KE.shape)
    #U = torch.cat(4*[U],dim=-1)
    #print(p_list)

    dqdt_predicted = grad(KE.sum(1), p_list, grad_outputs=torch.ones_like(KE.sum(1)), create_graph=True)[0] # dqdt = -dH/dp
    #print('dqdt_predicted',dqdt_predicted)

    return (dqdt_predicted, dpdt_predicted) # all data is arrange in q p manner