#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 18:07:07 2020

@author: simon
"""

import torch.nn as nn
from .derivative_estimator import derivative_ML
from datetime import datetime

class MLP2H_Separable_Hamil_VV(nn.Module):
    def __init__(self, n_input, n_hidden, n_stack = 1):
        '''
        VV : velocity verlet
        
        please check MLP2H_Separable_Hamil_LF.py for full documentation

        '''
        super(MLP2H_Separable_Hamil_VV,self).__init__()
        self.linear_kinetic = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, 1)
            )
        
        self.linear_potential = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, 1)
            )
        
        self.n_stack = n_stack
        
    def forward(self, q_list, p_list, time_step):
        '''
        Forward pass using velocity verlet, only for DIM = 1

        Parameters
        ----------
        q_list : torch.tensor of N X 1 
            tensor of position
        p_list : torch.tensor of N X 1 
            tensor of momentum
            
        time_step : float
            as described by dataset

        Returns
        -------
        q_list,p_list : torch.tensor 
            torch.tensor of (q_next,p_next) of N X 2 

        please check MLP2H_Separable_Hamil_LF.py for full documentation
        '''

        for i in range(self.n_stack) : # stack the NN

            dqdt_predicted, dpdt_predicted = derivative_ML(q_list, p_list, self.linear_potential, self.linear_kinetic)
            p_list = p_list + dpdt_predicted * time_step / 2
            # we need to use prediction again instead of using p_list because we need correction

            dqdt_predicted, dpdt_predicted = derivative_ML(q_list, p_list, self.linear_potential, self.linear_kinetic)
            q_list = q_list + dqdt_predicted * time_step# next time step

            dqdt_predicted, dpdt_predicted = derivative_ML(q_list, p_list , self.linear_potential, self.linear_kinetic)
            p_list = p_list + dpdt_predicted * time_step / 2

        #print('vv',q_list,p_list)
        return (q_list, p_list)
    
    def set_n_stack(self, n_stack:int):
        '''setter function for n stack'''
        self.n_stack = n_stack