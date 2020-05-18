#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 11:35:40 2020

@author: simon
"""

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset,TensorDataset,DataLoader
from torchvision import datasets,transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm,trange
from torch.autograd import grad 
import os
import numpy as np
from error import *

curr_path = os.path.dirname(__file__)

class MLP1H_KE(nn.Module):
    def __init__(self, n_input, n_hidden):
        super(MLP1H_KE,self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, 1) #since the output always 1 hamiltonian
            )
        
    def forward(self, p):
        return self.linear(p)

class MLP1H_Potential(nn.Module):
    def __init__(self, n_input, n_hidden):
        super(MLP1H_Potential,self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, 1) #since the output always 1 hamiltonian
            )
        
    def forward(self, q):
        return self.linear(q)
    
class MLP1H_General_Hamil(nn.Module):
    def __init__(self, n_input, n_hidden):
        super(MLP1H_General_Hamil,self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, 1) #since the output always 1 hamiltonian
            )
        
    def forward(self, p, q):
        states = torch.cat((p,q), axis = 1)
        return self.linear(states)
    
class MLP2H_General_Hamil(nn.Module):
    def __init__(self, n_input, n_hidden):
        super(MLP2H_General_Hamil,self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden,n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, 1) #since the output always 1 hamiltonian
            )
        
    def forward(self, p, q):
        states = torch.cat((p,q), axis = 1)
        return self.linear(states)
    
class MLP1H_Separable_Hamil(nn.Module):
    def __init__(self,n_input, n_hidden):
        super(MLP1H_Separable_Hamil,self).__init__()
        self.linear_kinetic = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden,1)
            )
        
        self.linear_potential = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden,1),
            )
        
    def forward(self,p,q):
        return self.linear_kinetic(p) + self.linear_potential(q)
    
class MLP2H_Separable_Hamil(nn.Module):
    def __init__(self, n_input, n_hidden):
        super(MLP2H_Separable_Hamil,self).__init__()
        self.linear_kinetic = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden,1)
            )
        
        self.linear_potential = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, 1)
            )
        
    def forward(self, p, q):
        # return self.linear_kinetic(p);
        return self.linear_kinetic(p) + self.linear_potential(q)
    
class MLP3H_Separable_Hamil(nn.Module):
    def __init__(self, n_input, n_hidden):
        super(MLP3H_Separable_Hamil,self).__init__()
        self.linear_kinetic = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden,1)
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
        
    def forward(self, p, q):
        return self.linear_kinetic(p) + self.linear_potential(q)