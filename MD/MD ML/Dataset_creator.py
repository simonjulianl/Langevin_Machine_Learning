#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 13:27:02 2020

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

#%%
def load_raw():
    currdir = os.getcwd()
    data_dir = currdir +  '/Initial'
    pos_files = sorted(glob.glob(data_dir + '/pos_sampled.npy'))
    vel_files = sorted(glob.glob(data_dir + '/velocity_sampled.npy'))

    assert len(pos_files) == len(vel_files)
   
    # there must be a 1-1 correspondence of position and velocity 
    
    final_pos = np.hstack([np.load(file) for file in pos_files])
    final_vel = np.hstack([np.load(file) for file in vel_files])
    
    return final_pos,final_vel

def DoWork(simulation):
    simulation.simulate(track = False)
    return simulation

class MD_dataset(Dataset):
    # =============================================================================
    '''     Data must be aranged in this order : 
        1) initial position
        2) Final data slower time step 
        3) Final data larger time step '''
    # =============================================================================
    def __init__(self,data):
        self.dataset = []
        initial_data = torch.tensor(data[0],dtype = torch.float)
        final_data_truth = torch.tensor(data[1] , dtype = torch.float)
        final_data_wrong = torch.tensor(data[2], dtype = torch.float)
        
        assert len(final_data_truth) == len(final_data_wrong)
  
        for i in range(len(final_data_truth)):
            #the data is arranged in the form of [[p1 q1]
            #                                     [p2 q2]] 
            #where  is the initial and 2 is the final
            data_pair = torch.stack((initial_data[i],final_data_wrong[i])) #this is pairwise initial and error
            # data_pair = initial_data[i] # this is only the initial state
            
            # self.dataset.append((data_pair,final_data_truth[i]))
            
            true_residue = final_data_truth[i] - final_data_wrong[i]
            self.dataset.append((data_pair,true_residue,final_data_truth[i]))
            
    def __len__(self):
         return len(self.dataset)
     
    def __getitem__(self,idx):
        return self.dataset[idx][0], self.dataset[idx][1], self.dataset[idx][2]
                    

#%%    
def simulate(total_time_slow = 5,slow_time_step = 0.05,jobid = 1, LSTM = False):
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
    N_validation, N_validation_current = 0.2 * len(init_pos),0
    
    i = 0
    while N_train_current < N_train:
        grid_train.append(combination[i])
        q,p = combination[i]
        N_train_current += len(grid[q][p])
        i += 1 
    
    while N_validation_current < N_validation:
        grid_validation.append(combination[i])
        q,p = combination[i]
        N_validation_current += len(grid[q][p])
        i += 1 
    
    grid_test = combination[i:]
    
    print('Actual Split : {:.4f} Train {:.4f} Validation {:.4f} Test'.format(
        100.*N_train_current / len(init_pos), 100. * N_validation_current / len(init_pos),
        (1 - N_train_current / len(init_pos) - N_validation_current / len(init_pos)) * 100.
        ))
    
    #write the data onto txt file
    with open('splitting_info.txt','w') as f:
        f.write('train : ')
        f.write(str(100.*N_train_current / len(init_pos)))
        f.write('\n')
        f.write('validation: ')
        f.write(str(100. * N_validation_current / len(init_pos)))
        f.write('\n')
        f.write('test : ')
        f.write(str((1 - N_train_current / len(init_pos) - N_validation_current / len(init_pos)) * 100.))
        f.write('\n')
        
    
    modes = ['train','validation','test']
 
    for mode in modes :
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
           
        elif mode == 'validation':
            N = N_validation_current
            for q,p in grid_validation:
                temporary = grid[q][p]
                for item in temporary : 
                    init_pos.append(item[0])
                    init_vel.append(item[1])
            init_pos = np.array(init_pos)
            init_vel = np.array(init_vel)
            
        else:
            N = total_particle - N_train_current - N_validation_current
            for q,p in grid_test:
                temporary = grid[q][p]
                for item in temporary : 
                    init_pos.append(item[0])
                    init_vel.append(item[1])
            init_pos = np.array(init_pos)
            init_vel = np.array(init_vel)
            
        total_time_slow = total_time_slow
        slow_time_step = slow_time_step
        time_step_multiplier = total_time_slow # we define it such that the fast always take 1 step
        
        total_time_fast = int(total_time_slow / time_step_multiplier)
        fast_time_step = slow_time_step * time_step_multiplier
        
        pool = Pool(processes = 2)
        
        simulation_slow = Simulation_PQP(N = N, DIM =1, BoxSize =1.0, deltat = slow_time_step, Nsteps = 90)
        #change this if necessary since I force to 90 time steps only
        simulation_slow.set_mass(1)
        simulation_slow.set_scaling(True)
        simulation_slow.initialize_pos(scale = 2,file =False)
        simulation_slow.initialize_vel(scale = 1, v_scaling = True)
        
        simulation_fast = Simulation_PQP(N = N, DIM = 1, BoxSize = 1.0, deltat = slow_time_step * 10, Nsteps = total_time_slow // 10)
        simulation_fast.set_mass(1)
        simulation_fast.set_scaling(True)
        simulation_fast.initialize_pos(scale = 2 ,file =False)
        simulation_fast.initialize_vel(scale = 1, v_scaling = False)
        
        assert len(simulation_slow.pos) == N == len(simulation_fast.pos)
        
        for i in range(len(simulation_slow.pos)):
            simulation_slow.pos[i] = init_pos[i]
            simulation_slow.vel[i] = init_vel[i]
            
            simulation_fast.pos[i] = init_pos[i]
            simulation_fast.vel[i] = init_vel[i]
            
        simulation = (simulation_slow,simulation_fast)
            
        simulation_slow,simulation_fast = pool.map(DoWork,simulation)
        
        pool.close()
        pool.join()
        
        final_true_pos = simulation_slow.pos
        final_true_vel = simulation_slow.vel
        
        final_wrong_pos = simulation_fast.pos
        final_wrong_vel = simulation_fast.vel
        
        print('\n displaying the difference between fast and slow')
        for i in range(10):
            print('initial : {:.4f} \t truth : {:.4f} \t wrong : {:.4f}'.format(
                float(init_pos[i]),
                float(final_true_pos[i].item()),
                float(final_wrong_pos[i].item())
                ))
               
        if LSTM: 
            #only consider the slow one since the fast only has 1 time step 
            currdir = os.getcwd()
            true_pos_track = np.array(simulation_slow.xlist_all)[:,:-1] #since we are dealing with 1D particle 
            true_vel_track = np.array(simulation_slow.vxlist_all)[:,:-1]
            last_pos = np.array(simulation_slow.xlist_all)[:,-1]
            last_vel = np.array(simulation_slow.vxlist_all)[:,-1]
            np.save(currdir + '/compiled/true_pos_track_{}{}.npy'.format(mode,jobid),true_pos_track)
            np.save(currdir + '/compiled/true_vel_track_{}{}.npy'.format(mode,jobid),true_vel_track)
            np.save(currdir + '/compiled/final_pos_track_{}{}.npy'.format(mode,jobid),last_pos)
            np.save(currdir + '/compiled/final_vel_track_{}{}.npy'.format(mode,jobid),last_vel)
        
        else:
            currdir = os.getcwd()
            # np.save(currdir + '/compiled/initial_pos_{}{}.npy'.format(mode,jobid),init_pos)
            # np.save(currdir + '/compiled/initial_vel_{}{}.npy'.format(mode,jobid),init_vel)
            # np.save(currdir + '/compiled/final_true_pos_{}{}.npy'.format(mode,jobid),final_true_pos)
            # np.save(currdir + '/compiled/final_true_vel_{}{}.npy'.format(mode,jobid),final_true_vel)
            # np.save(currdir + '/compiled/final_wrong_pos_{}{}.npy'.format(mode,jobid),final_wrong_pos)
            # np.save(currdir + '/compiled/final_wrong_vel_{}{}.npy'.format(mode,jobid),final_wrong_vel)
            
            true_pos_track = np.array(simulation_slow.xlist_all) #since we are dealing with 1D particle 
            true_vel_track = np.array(simulation_slow.vxlist_all)
            
            cont_pos = true_pos_track[:,-1]
            cont_vel = true_vel_track[:,-1]
            
            # np.save(currdir + '/compiled/true_pos_track_{}{}.npy'.format(mode,jobid),true_pos_track)
            # np.save(currdir + '/compiled/true_vel_track_{}{}.npy'.format(mode,jobid),true_vel_track)
            
            #continuation of integration
            pool = Pool(processes = 2)
            
            Rest_Nsteps = 2 # this is by default, should adjust
            simulation_slow = Simulation_PQP(N = N, DIM =1, BoxSize =1.0, deltat = slow_time_step, Nsteps = Rest_Nsteps)
            simulation_slow.set_mass(1)
            simulation_slow.set_scaling(True)
            simulation_slow.initialize_pos(scale = 2,file =False)
            simulation_slow.initialize_vel(scale = 1, v_scaling = True)
         
            print('RestNsteps :,' ,Rest_Nsteps)
            print()
            print('false time step : ', slow_time_step * Rest_Nsteps)
            simulation_fast = Simulation_PQP(N = N, DIM = 1, BoxSize = 1.0, deltat = slow_time_step * Rest_Nsteps, Nsteps = 1)
            simulation_fast.set_mass(1)
            simulation_fast.set_scaling(True)
            simulation_fast.initialize_pos(scale = 2 ,file =False)
            simulation_fast.initialize_vel(scale = 1, v_scaling = False)
            
            for i in range(len(simulation_slow.pos)):
                simulation_slow.pos[i] = cont_pos[i]
                simulation_slow.vel[i] = cont_vel[i]
                
                simulation_fast.pos[i] = cont_pos[i]
                simulation_fast.vel[i] = cont_vel[i]
                
            assert (simulation_fast.pos == simulation_slow.pos).all()
            assert (simulation_fast.vel == simulation_slow.vel).all()
            
            simulation = (simulation_slow,simulation_fast)
            
            simulation_slow,simulation_fast = pool.map(DoWork,simulation)
            
            pool.close()
            pool.join()
            
            rest_true_pos_track = np.array(simulation_slow.xlist_all) #since we are dealing with 1D particle 
            rest_true_vel_track = np.array(simulation_slow.vxlist_all)
            
            true_pos_track = np.concatenate((true_pos_track, rest_true_pos_track[:,1:]), axis = 1)
            true_vel_track = np.concatenate((true_vel_track, rest_true_vel_track[:,1:]), axis = 1)
            false_last_pos = np.array(simulation_fast.xlist_all)[:,-1]
            false_last_vel = np.array(simulation_fast.vxlist_all)[:,-1]
                       
            np.save(currdir + '/compiled/true_pos_track_{}{}.npy'.format(mode,jobid),true_pos_track)
            np.save(currdir + '/compiled/true_vel_track_{}{}.npy'.format(mode,jobid),true_vel_track)
            np.save(currdir + '/compiled/false_last_pos_{}{}.npy'.format(mode,jobid),false_last_pos)
            np.save(currdir + '/compiled/false_last_vel_{}{}.npy'.format(mode,jobid),false_last_vel)
                
def create_dataset(mode = 'train',total_time_slow = 5,slow_time_step = 0.05, plot = False):
    # =============================================================================
    # The simulation uses asymmetrical double well external force and no interaction between particles    
    # =============================================================================
    '''There are 3 modes : train, validation and test with 60 20 20 % splitting 
    Use JOBID = 1 for dataset linear / convolutional MD ML
    Use JOBID = 2 for dataset for LSTM 
    '''
    
    currdir = os.getcwd() + '/compiled/'

    if not os.path.exists(currdir + 'initial_pos_{}1.npy'.format(mode)) : 
        simulate(total_time_slow, slow_time_step, jobid = 1)
    init_pos = np.load(currdir +'initial_pos_{}1.npy'.format(mode))
    init_vel = np.load(currdir +'initial_vel_{}1.npy'.format(mode))
    final_true_pos = np.load(currdir +'final_true_pos_{}1.npy'.format(mode))
    final_true_vel = np.load(currdir +'final_true_vel_{}1.npy'.format(mode))
    final_wrong_pos = np.load(currdir +'final_wrong_pos_{}1.npy'.format(mode))
    final_wrong_vel = np.load(currdir +'final_wrong_vel_{}1.npy'.format(mode))
                
    qtick = np.arange(-2,2 + 0.5,0.5)
    ptick = np.arange(-4.0,4.5 + 0.5,0.5)
   
    if plot : 
        fig = plt.figure(figsize = (10,15))
        
        ax = fig.add_subplot(311)
        ax.set_title('Old configuration')
        ax.set_ylabel('p (momentum) ')
        ax.set_xticks(qtick)
        ax.set_yticks(ptick)
        ax.set_ylim(-4,4)
        ax.set_xlim(-2,2)
        ax.set_xlabel('q (position) ')
        ax.grid(True)
        assert len(init_pos) == len(init_vel)
        
        for i in range(len(init_pos)):
            ax.scatter(init_pos[i],init_vel[i], c ='k', marker = 'o', s = 1)
     
        ax = fig.add_subplot(312)
        ax.set_title('New Configuration True')
        ax.set_ylabel('p (momentum) ')
        ax.set_xlabel('q (position) ')
        ax.set_xticks(qtick)
        ax.set_yticks(ptick)
        ax.set_ylim(-4,4)
        ax.set_xlim(-2,2)
        ax.grid(True)
        assert len(final_true_pos) == len(final_true_vel)
        
        for i in range(len(final_true_pos)):
            ax.scatter(final_true_pos[i],final_true_vel[i], c ='k', marker = 'o' , s = 1)
            
        ax = fig.add_subplot(313)
        ax.set_title('New Configuration Wrong')
        ax.set_ylabel('p (momentum) ')
        ax.set_xlabel('q (position) ')
        ax.set_xticks(qtick)
        ax.set_yticks(ptick)
        ax.set_ylim(-4,4)
        ax.set_xlim(-2,2)
        ax.grid(True)
        assert len(final_wrong_pos) == len(final_wrong_vel)
        
        for i in range(len(final_wrong_pos)):
            ax.scatter(final_wrong_pos[i],final_wrong_vel[i], c ='k', marker = 'o' , s = 1)
            
        plt.show()
 
    init_data = np.concatenate((init_pos,init_vel),axis = 1)
    final_data_true = np.concatenate((final_true_pos,final_true_vel), axis = 1)
    final_data_wrong = np.concatenate((final_wrong_pos,final_wrong_vel), axis = 1)

    data = (init_data,final_data_true,final_data_wrong)
    dataset = MD_dataset(data)
    
    return dataset