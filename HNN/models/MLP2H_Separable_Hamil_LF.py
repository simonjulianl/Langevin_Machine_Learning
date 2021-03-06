#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 15:46:31 2020

@author: simon
"""

import torch.nn as nn
from .derivative_estimator import derivative_ML

'''list of models available for training
based on previous result, 1D particles of double well seems to perform best for
2 hidden layer, 20 units per hidden layer '''

class MLP2H_Separable_Hamil_LF(nn.Module):
    def __init__(self, n_input, n_hidden, n_stack = 1):
        '''
        Chained Modified Hamiltonian Neural Network
        given q0,p0 the data flow is 
        q0, p0 --> NN --> q1, p1 --> NN --> q2, p2 --> and so on
        this allows for longer training steps with weight sharing between each stacked NN 

        LF : Leapfrog algorithm
        
        Parameters
        ----------
        n_input : int
            number of input dimensions/channel
        n_hidden : int
            number of neurons per hidden layer
        n_stack : int, optional
            Number of stacked NN. The default is 1.
            
        Precaution
        ----------
        When loading model, please set torch.manual_seed due to initialization process

        '''
        super(MLP2H_Separable_Hamil_LF,self).__init__()
        self.linear_kinetic = nn.Sequential(
            nn.Linear(n_input, n_hidden),
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
            nn.Linear(n_hidden, 1)
            )
        
        self.n_stack = n_stack
        
    def forward(self, q_list, p_list, time_step):
        '''
        forward pass using leap frog only for DIM = 1

        Parameters
        ----------
        q_list : torch.tensor of N X 1 
            tensor of position
        p_list : torch.tensor of N X 1 
            tensor of momentum
                    
        time_step : time step for 1 update
            time_step of the integration as described by dataset

        Returns
        -------
        q_list, p_list : tuple
            predicted q and p of next time_step, here is akin to using leapfrog

        Precaution
        ----------
        Strictly speaking, the potential and kinetic produce hamiltonian by itself,
        However since we want to chain it, the output becomes the next time step and it doesnt 
        produce hamiltonian anymore,
        
        to get the approximate hamiltonian function
        use class.linear_kinetic and class.linear_potential and then torch.load_state_dict 
        '''
        
        for i in range(self.n_stack) : # stack the NN 
            dqdt_predicted, dpdt_predicted = derivative_ML(q_list, p_list, self.linear_potential, self.linear_kinetic)
            q_list = q_list + dqdt_predicted * time_step# next time step 
            dqdt_predicted, dpdt_predicted = derivative_ML(q_list, p_list, self.linear_potential, self.linear_kinetic)
            p_list = p_list + dpdt_predicted * time_step
            
        return (q_list, p_list)
    
    def set_n_stack(self, n_stack:int):
        '''setter function for n stack'''
        self.n_stack = n_stack