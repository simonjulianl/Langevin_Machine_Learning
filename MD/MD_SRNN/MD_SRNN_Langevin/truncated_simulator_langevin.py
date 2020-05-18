#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 14:31:41 2020

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

import os
from pathlib import Path

import sys
curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
initialization = os.path.join(parent_path, 'initial')

sys.path.append(parent_path)
from error.error import *;
from tqdm import trange;
from models import *;

np.random.seed(2) # change the seed to large number
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class simulation_langevin:
    __instance = None;
    kB = 1
    def __init__(self, time_step : float, Nsteps : int, gamma : float, Temperature : float, Func = None, **kwargs):
        try : 
            print('Initializing')
            self.time_step = time_step
            self.Nsteps = Nsteps
            self.gamma = gamma 
            self.Temp = Temperature
            self.Func = Func
            self.load()
            self.randNormal()
            
            if simulation_langevin.__instance != None:
                raise SingletonError('This class is a singleton')
            else :
                simulation_langevin.__instance = self
            
        except InitializationError:
            raise InitializationError('could not initialize')
            
        
        print('Initialization Finished')
        
    def load(self) -> None:
        ''' helper function to load the data, since we are going to get the p and q 
        distribution overtime, 100 independent trajectories should be enough'''
        self.q = np.load(initialization + '/pos_sampled.npy')[:100].squeeze();
        self.p = np.load(initialization + '/velocity_sampled.npy')[:100].squeeze();
        
    def change(self, time_step : float, Nsteps : int, gamma : float, Temperature : float, Func = None) :
        try : 
            print('Changing setting')
            self.time_step = time_step
            self.Nsteps = Nsteps
            self.gamma = gamma 
            self.Temp = Temperature
            self.Func = Func
            self.load()            
        except InitializationError:
            raise InitializationError('could not initialize')
            
        print('Setting changed')
    
    def randNormal(self): 
        '''helper function to set the random for langevin
        using BP method, there are 2 random vectors'''
        random_1 = np.random.normal(loc = 0.0, scale = 1.0, size = 200000)
        random_2 = np.random.normal(loc = 0.0, scale = 1.0, size = 200000)
        self.random_1 = random_1.reshape(-1,100)
        self.random_2 = random_2.reshape(-1,100)
            
    def integrate(self) -> tuple: # of (q list ,  p list)
        self.load(); #set the p and q
        ''' mass counted as 1 hence omitted '''
        idx = 0 # counter for random, reset every 1000 steps if used
                
        q_list = np.zeros((self.Nsteps + 1,100))
        p_list = np.zeros((self.Nsteps + 1,100))
        
        q_list[0] = self.q
        p_list[0] = self.p
        
        for i in trange(1, self.Nsteps+1):
            self.p = np.exp(-self.gamma * self.time_step / 2) * self.p + np.sqrt(self.kB * self.Temp * ( 1 - np.exp( - self.gamma * self.time_step))) * self.random_1[idx]
            
            acc = self.get_force()

            self.p = self.p + self.time_step / 2 * acc #dp/dt
            
            self.q = self.q + self.time_step * self.p
            
            acc = self.get_force()
            self.p = self.p + self.time_step / 2 * acc
            
            self.p = np.exp(-self.gamma * self.time_step / 2) * self.p + np.sqrt(self.kB * self.Temp * ( 1 - np.exp( - self.gamma * self.time_step))) * self.random_2[idx]
            
            q_list[i] = self.q
            p_list[i] = self.p
            idx += 1
            
            if (idx == 2000):
                # self.randNormal() # generate again
                idx = 0 # reset counter
            
        
        return (q_list.T, p_list.T) # transpose to get particle x trajectory 
    
    
    def get_force(self):
        ''' double well potential and force manually computed 
        just simple code for 1 Dimension without generalization'''

        acc = np.zeros([100])
        for i in range(100):
            q = self.q[i]
            phi = (q ** 2.0 - 1) ** 2.0 + q # potential
            dphi = (4 * q) * (q ** 2.0 - 1) + 1.0 # force = dU / dq
            acc[i] = acc[i] - dphi
            

        return acc
    