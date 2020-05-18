#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 21:14:48 2020

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

import glob 
import os
from pqp_integrator import Simulation_PQP 
from multiprocessing import Pool
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import product
import random
from Dataset_creator import load_raw 

def DoWork(simulation): #helper function for multiprocessing
    simulation.simulate(track = False)
    return simulation

'''API : 
    To use, simulate(Nsteps, deltat, jobid)
    where Nsteps is the total number of steps taken by the simulation
    deltat is the accurate time step 
    
    return :
        -train data
        -validation data
        -integrated validation data
'''
        
def simulate(Nsteps = 100,slow_time_step = 0.001):
    init_pos,init_vel = load_raw()

    def nested_dict(n, type):
        if n == 1:
            return defaultdict(type)
        else:
            return defaultdict(lambda : nested_dict(n-1,type))
        
    qtick = np.arange(-2 -0.5 ,2 + 0.5,0.5)
    ptick = np.arange(-4.0 - 0.5,4.5 + 0.5,0.5)
    grid = nested_dict(2,list)
    print('Total Grid : {}'.format(len(qtick) * len(ptick)))
    for x in qtick:
        for y in ptick:
            grid[x][y] = []
            
    qlist = list(grid.keys())
    plist = list(ptick)
 
    for position, momentum in zip(init_pos,init_vel):
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
    
    assert total_particle == len(init_pos)
    
    #randomly choose the grid until N ~ 60 % 20 % 20 % split
    combination = list(product(qlist,plist))
    for i in range(10):
        combination = random.sample(combination,len(combination))
        
    grid_train, grid_validation = [], [] #coordinate of grids for training
    N_train, N_train_current = 0.6 * len(init_pos),0
    
    i = 0
    while N_train_current < N_train:
        grid_train.append(combination[i])
        q,p = combination[i]
        N_train_current += len(grid[q][p])
        i += 1 
    
    
    grid_validation = combination[i:]
    
    print('Actual Split : {:.4f} Train {:.4f} Validation '.format(
        100.*N_train_current / len(init_pos), 100. * (total_particle - N_train_current) / len(init_pos)
        ))
    
    
    modes = ['train','validation']
 
    for mode in modes :
        print("Current mode : {}".format(mode))
        
        init_pos = []
        init_vel = []
        if mode == 'train':
            N = N_train_current
            for q,p in grid_train:
                temporary = grid[q][p]
                for item in temporary : 
                    init_pos.append(item[0])
                    init_vel.append(item[1])
            init_pos = np.array(init_pos)
            init_vel = np.array(init_vel)
           
        else :
            N = total_particle - N_train_current
            for q,p in grid_validation:
                temporary = grid[q][p]
                for item in temporary : 
                    init_pos.append(item[0])
                    init_vel.append(item[1])
            init_pos = np.array(init_pos)
            init_vel = np.array(init_vel)
            
                        
        pool = Pool(processes = 2)
        
        simulation_fast = Simulation_PQP(N = N, DIM =1, BoxSize =1.0, deltat = 0.001, Nsteps = 1000, DumpFreq = 100)
        #change this if necessary since I force to 90 time steps only
        simulation_fast.set_mass(1)
        simulation_fast.set_scaling(True)
        simulation_fast.initialize_pos(scale = 2,file =False)
        simulation_fast.initialize_vel(scale = 1, v_scaling = True)
        
        simulation_slow = Simulation_PQP(N = N, DIM =1, BoxSize =1.0, deltat = slow_time_step, Nsteps = int(Nsteps), DumpFreq = 100)
        #change this if necessary since I force to 90 time steps only
        simulation_slow.set_mass(1)
        simulation_slow.set_scaling(True)
        simulation_slow.initialize_pos(scale = 2,file =False)
        simulation_slow.initialize_vel(scale = 1, v_scaling = True)

        for i in range(len(simulation_slow.pos)):
            #initialize the slow position
            simulation_slow.pos[i] = init_pos[i]
            simulation_slow.vel[i] = init_vel[i]
            
            simulation_fast.pos[i] = init_pos[i]
            simulation_fast.vel[i] = init_vel[i]

        #simulation fast just take the first 5000 steps of the simulation slow       
        simulation = (simulation_slow, simulation_fast)
        
        simulation_slow,simulation_fast = pool.map(DoWork,simulation) #to take the first object of the simulation
        
        pool.close()
        pool.join()
        
        #the data is arranged from t0 to tmax hence there are Nsteps + 1
        #assuming we have deltat = 0.001 and we want to match with 0.5, hence we slice every 500 data 
        #hence we will have t0 t500 t1000 ... remember than t1000 is equal when t = 1000 * 0.001 = 1.0
        xlist = np.array(simulation_slow.xlist_all)[:,:]
        vxlist = np.array(simulation_slow.vxlist_all)[:,:]
        
        currdir = os.getcwd()
        
        #save the highly accurate reference point for t = 0.001
        np.save(currdir + '/compiled/accurate_pos_track01.npy', xlist)
        np.save(currdir + '/compiled/accurate_vel_track01.npy',vxlist)
        
        del xlist, vxlist
        #truncate the data to obtain for ML using the larger timestep
        
        xlist_data = np.array(simulation_fast.xlist_all)[:,:] # + 1 because of the initial position is calculated
        vxlist_data = np.array(simulation_fast.vxlist_all)[:,:]
        
        simulation_cont_fast = Simulation_PQP(N = N, DIM = 1, BoxSize = 1.0, deltat = 0.5 , Nsteps = 1)
        #change this if necessary since I force to 90 time steps only
        simulation_cont_fast.set_mass(1)
        simulation_cont_fast.set_scaling(True)
        simulation_cont_fast.initialize_pos(scale = 2,file =False)
        simulation_cont_fast.initialize_vel(scale = 1, v_scaling = True)
        
        for i in range(len(simulation_slow.pos)):
            #initialize the cont fast to get the deltat = 0.5
            #use the second last data since the last data is used as ground truth t100
            simulation_cont_fast.pos[i] = xlist_data[i][-2]
            simulation_cont_fast.vel[i] = vxlist_data[i][-2]
            
        #integrate the timestep
        simulation_cont_fast.simulate()
        
        xlist_wrong = np.array(simulation_cont_fast.xlist_all)[:,-1]
        vxlist_wrong = np.array(simulation_cont_fast.vxlist_all)[:,-1]
        
        
        x_data = np.concatenate((xlist_data[:,:-1],np.expand_dims(xlist_wrong,axis = 1)), axis = 1)
        vx_data = np.concatenate((vxlist_data[:,:-1],np.expand_dims(vxlist_wrong,axis = 1)), axis = 1)
        data = np.concatenate((np.expand_dims(x_data,axis = 1), np.expand_dims(vx_data, axis = 1)), axis = 1)
        #data is arranged in n x 2 x 11 hence there are 2 channels 10 true state and 1 false state
        
        true_state = np.concatenate((np.expand_dims(xlist_data[:,-1],axis = 1),np.expand_dims(vxlist_data[:,-1], axis = 1)), axis = 1)
        
        #data is always arranged in q p arrangement, meaning the position goes in the left and momentum on the right
        
        np.save(currdir + '/compiled/ml_data_{}_01.npy'.format(mode),data)
        np.save(currdir + '/compiled/ml_truestate_{}_01.npy'.format(mode), true_state)
    
            
if __name__ == "__main__":
    #since deltat = 0.001, to achieve total t = 10 , Nsteps = 10000
    simulate(500, 0.001)