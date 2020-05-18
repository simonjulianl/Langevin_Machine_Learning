#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 10:07:47 2020

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

from error.error import *
from models import *
from collections import defaultdict
from itertools import product
import random

curr_path = os.path.dirname(__file__)

np.random.seed(937162211)
random.seed(937162211)
class simulator_numerical:
    '''time_step : float, Nsteps : int , DumpFreq : int
    parameters : 
        time_step / tau : float
        Nsteps : int
        DumpFreq : 1 // how often we store the value of q and p with respect to Nsteps
        
        by default, N (total particle) = 25 000 
        there is no periodic boundary condition since we are using
        potential barrier at x= 2 and x= -2 , Mass and Temp = 1 for simplicity
        all the particles is done in 1 Dimension for simplicity'''
    
    #helper function to load all the initial states
    def initialize(self) -> torch.tensor :
        # we should mix a temperature here, I have defined from T = 1 , ... , 10 generated
        Temp = [1, 2, 3, 4, 5, 6, 7 , 8 ,9, 10]
        initial_q = np.array([])
        initial_p = np.array([])
        for T in Temp : 
            path_q = os.path.join(curr_path + '/initial/pos_sampled_T{}.npy'.format(T))
            path_p = os.path.join(curr_path + '/initial/velocity_sampled_T{}.npy'.format(T))
            q_temp = np.load(path_q)[:2500]
            p_temp = np.load(path_p)[:2500]
            
            if len(initial_q) == 0 :
                initial_q = q_temp
                initial_p = p_temp
            else :
                initial_q = np.concatenate((initial_q, q_temp))
                initial_p = np.concatenate((initial_p, p_temp))
 
        #Suggested to mix the dataset at different temperature 
        #one Way to do it is to plot the pdf graph and ignore the very rare events where the pdf falls below 0.0001
        #while using random dataset
        # initial_q = np.random.uniform(-2,2,100000).reshape(100000,1)
        # initial_p = np.random.uniform(-4,4,100000).reshape(100000,1) # 4 is obtained from max and min of MCMC
        
        initial_q = initial_q.reshape(initial_q.shape[0], 1)
        initial_p = initial_p.reshape(initial_p.shape[0], 1)
        assert len(initial_q) == len(initial_p)
        N = initial_q.shape[0]
        
        ''''the initialization always take q p for each state !'''
        
        def nested_dict(n, type):
            if n == 1:
                return defaultdict(type)
            else:
                return defaultdict(lambda : nested_dict(n-1,type))
            
        qtick = np.arange(np.min(initial_q) -0.5 ,np.max(initial_q) + 0.5,0.5)

        ptick = np.arange(np.min(initial_p) - 0.5, np.max(initial_p) + 0.5,0.5)
        grid = nested_dict(2,list)
        print('Total Grid : {}'.format(len(qtick) * len(ptick)))
        for x in qtick:
            for y in ptick:
                grid[x][y] = []
                
        qlist = list(grid.keys())
        plist = list(ptick)
     
        for position, momentum in zip(initial_q,initial_p):
            lower_bound_pos, lower_bound_momentum = None, None
            i = 1 #iterate through the qlist first 
            while lower_bound_pos is None:
                if position.item() < qlist[i]:
                    # print(qlist[i])
                    lower_bound_pos = qlist[i-1]
                i += 1
                
            i = 1 # reset counter
            while lower_bound_momentum is None:
                if momentum.item() < plist[i] :
                    lower_bound_momentum = plist[i-1]
                i += 1
            grid[lower_bound_pos][lower_bound_momentum].append((position,momentum))
            
        total_particle = 0
        for x in qtick:
            for y in ptick:
                total_particle += len(grid[x][y])
                # print(x,y,len(grid[x][y]))
                
        #randomly choose the grid until N ~ 60 % 20 % 20 % split
        combination = list(product(qlist,plist))
        for i in range(10):
            combination = random.sample(combination,len(combination))
            
        grid_train, grid_validation = [], [] #coordinate of grids for training
        N_train, N_train_current = 0.6 * len(initial_q),0
        
        i = 0
        while N_train_current < N_train:
            grid_train.append(combination[i])
            q,p = combination[i]
            N_train_current += len(grid[q][p])
            i += 1 
        
        
        grid_validation = combination[i:]
        
        print('Actual Split : {:.4f}% Train / {:.4f}% Validation '.format(
            100.*N_train_current / N, 100. * (total_particle - N_train_current) / N
            ))
        
        
        modes = ['train','validation']
        
        for mode in modes :
            init_pos = []
            init_vel = []
            if mode == "train" :
                for q,p in grid_train:
                    temporary = grid[q][p]
                    for item in temporary : 
                        init_pos.append(item[0])
                        init_vel.append(item[1])
                init_pos = np.array(init_pos)
                init_vel = np.array(init_vel)
        
                self.train_init_q = init_pos
                self.train_init_p = init_vel
            else :
                for q,p in grid_validation:
                    temporary = grid[q][p]
                    for item in temporary : 
                        init_pos.append(item[0])
                        init_vel.append(item[1])
                init_pos = np.array(init_pos)
                init_vel = np.array(init_vel)
                
                self.validation_init_q = init_pos
                self.validation_init_p = init_vel
            
    __instance = None;
    
    def new(self, time_step : float, Nsteps : int, DumpFreq : int = 1, *args, **kwargs):
        if simulator_numerical.__instance != None: #reset p and q
            self.q_train = self.train_init_q
            self.p_train = self.train_init_p
            
            self.q_validation = self.validation_init_q
            self.p_validation = self.validation_init_p
        else :
            self.initialize()
            
        self.time_step = time_step
        self.Nsteps = Nsteps
        self.DumpFreq = DumpFreq
        self.DIM = kwargs.get('DIM',1); # by default DIM is one
 
        self.N_train = self.train_init_q.shape[0]
        self.N_validation = self.validation_init_q.shape[0]
    
        self.q_train = self.train_init_q
        self.p_train = self.train_init_p
        
        self.q_validation = self.validation_init_q
        self.p_validation = self.validation_init_p
        
    def __init__(self, time_step : float, Nsteps : int, DumpFreq : int = 1, *args, **kwargs):
        
        try :
            print('initializing')
            self.new(time_step, Nsteps, DumpFreq, *args, **kwargs)
        except : 
            raise InitializationError('Unable to initialize')
            
        if simulator_numerical.__instance != None:
            raise SingletonError('This class is a singleton')
        else:
            simulator_numerical.__instance = self;

            
        # print('initialization complete')
        
    def change(self, time_step : float, Nsteps : int, DumpFreq : int = 1, *args, **kwargs) -> None:
        
        try : 
            self.time_step = time_step
            self.Nsteps = Nsteps
            self.DumpFreq = DumpFreq
            self.new(time_step, Nsteps, DumpFreq, *args, **kwargs)
        except : 
            raise InitializationError('Fail to change setting')

    
    def get_force(self, mode):
        ''' double well potential and force manually computed '''
        if mode =="train":
            acc = np.zeros([self.N_train, self.DIM])
            for i in range(self.N_train):
                for k in range(self.DIM):
                    q = self.q_train[i][k]
                    phi = (q ** 2.0 - 1) ** 2.0 + q # potential
                    dphi = (4 * q) * (q ** 2.0 - 1) + 1.0 # force = dU / dq
                    acc[i,k] = acc[i,k] - dphi
            
        else : 
            acc = np.zeros([self.N_validation, self.DIM])
            for i in range(self.N_validation):
                for k in range(self.DIM):
                    q = self.q_validation[i][k]
                    phi = (q ** 2.0 - 1) ** 2.0 + q # potential
                    dphi = (4 * q) * (q ** 2.0 - 1) + 1.0 # force = dU / dq
                    acc[i,k] = acc[i,k] - dphi
        return acc
   
    def velocity_verlet_numerical(self):
        #since mass = 1, p = vel
        acc = self.get_force(mode = "train")
        print(self.p_train.shape)
        self.p_train = self.p_train + self.time_step / 2 * acc #dp/dt
        
        self.q_train = self.q_train + self.time_step * self.p_train #dq/dt
        
        acc = self.get_force(mode = "train")
        self.p_train = self.p_train + self.time_step / 2 * acc #dp/dt
        
        #validation part
        acc = self.get_force(mode = "validation")
        self.p_validation = self.p_validation + self.time_step / 2 * acc #dp/dt
        
        self.q_validation = self.q_validation + self.time_step * self.p_validation #dq/dt
        
        acc = self.get_force(mode = "validation")
        self.p_validation = self.p_validation + self.time_step / 2 * acc #dp/dt
        
    def integrate(self):
        print('start integrating')
        try :
            q_list_train = self.q_train
            p_list_train = self.p_train
            
            q_list_validation = self.q_validation
            p_list_validation = self.p_validation
            
            #the 0-index is always the initial q and p
            for i in trange(self.Nsteps):
                self.velocity_verlet_numerical()
                if np.isnan(np.sum(self.q_train)) or np.isnan(np.sum(self.p_train)) or np.isnan(np.sum(self.p_validation)) or np.isnan(np.sum(self.q_validation))  :
                    raise IntegrationError
                    
                if ( i % self.DumpFreq == 0) :
                    q_list_train = np.concatenate((q_list_train, self.q_train), axis = 1)
                    p_list_train = np.concatenate((p_list_train, self.p_train), axis = 1)
                    q_list_validation = np.concatenate((q_list_validation, self.q_validation), axis = 1)
                    p_list_validation = np.concatenate((p_list_validation, self.p_validation), axis = 1)
        
        except IntegrationError:
            raise IntegrationError('Overflow / Value Exploding, Integration is unstable & time step is too big ')
            
        #return qlist and plist
                
        print('finish integrating')
        return ((q_list_train, p_list_train),(q_list_validation, p_list_validation))
    
    def __repr__(self):
        return 'Simulation velocity verlet \n \
        Force : Double Well Asymmetrical \n \
        Total Particle : {} \n \
        Total Training : {} \n \
        Total Validation : {} \n \
        Total Steps : {} \n \
        Dumping Frequency : {} \n \
        Time Step : {}'.format(self.N_train + self.N_validation, self.N_train, self.N_validation, self.Nsteps, self.DumpFreq, self.time_step)
        
# def initialize():
#     path_q = os.path.join(curr_path + '/initial/pos_sampled.npy')
#     path_p = os.path.join(curr_path + '/initial/velocity_sampled.npy')
#     initial_q = np.load(path_q)
#     initial_p = np.load(path_p)
    
#     states = np.concatenate((initial_q,initial_p), axis = 1)
    
#     ''''the initialization always take q p for each state !'''
    
#     return torch.tensor(states)
        
if __name__ == '__main__':
    model = MLP_General_Hamil(2, 10)


    new = simulator_numerical(0.1,Nsteps = 1, DumpFreq = 1)
    training, validation = new.integrate()
    print(training[0].shape)
    print(new)

    q = torch.Tensor(q_list[:,-1]).unsqueeze(1)
    p = torch.Tensor(p_list[:,-1]).unsqueeze(1)
    
    # velocity_verlet_ML(q,p, model, 0.01)
