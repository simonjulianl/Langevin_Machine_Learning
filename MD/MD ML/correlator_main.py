#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 15:19:40 2020

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
from correlator_MD import MDModel
import os
from correlator_dataset import create_dataset
from pqp_integrator import Simulation_PQP 

''' Files needed : 
    custom_layers
    correlator_data_generator
    correlator_dataset : create_dataset(mode, total_time)
        1) total time is calculated with respect to tau = 0.001 for reference
            hence total time 0.001 means 
            deltat is fixed at 0.001 as reference and the data is fed in t0 t0.5 t1.0 manner 
            so in total there are
            t0 t0.5 t1.0 t1.5 t2.0 t2.5 t3.0 t3.5 t4.0 t4.5 + t5.0' and predicts t5.0 residue
    simulator_pqp
    correlator MD : load the path, already created, run it from this file
    '''
    
# =============================================================================
#     Hyperparameter tuning
# =============================================================================
currdir = os.getcwd()
batch_size = 128
total_time = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MDModel().to(device)
model.load_state_dict(torch.load(currdir + '/compiled/MD_ML_conv_lasso_energy_loss_01.pth'))

''' the correlation here is the sum of q and p '''

def energy(state):
    return np.mean((state[:,0] ** 2 -1) ** 2 + state[:,0] + ( 0.5 * state[:,1] ** 2),axis = 0)

def process(reference_list, total_time):
    '''process is the function to create the history from using 
    1 ) the step ==> given 10 + 1 states, predict the 11th state
    2 ) After getting the 11th state, take from 2nd to 11th and false 12 ==> predict residue
    repeat the process until tmax
        -> total time is calculated based on 0.5 tau , hence if we have total_time = 5
        -> we would do 10 more time prediction and correction
    '''
    
    history_list = reference_list[:,:,:-1] # this will store n x 2 x .. as required by tmax and the last one which is the false is not included
    
    #since we are using fix timestep of 0.5 as trained
    # assert total_time % 0.5 == 0 and total_time >= 0.5
    
    current_input = reference_list
    
    N = len(current_input) # this is total particle
    for _ in range(int(total_time / 0.1)):
        temp = None; #this will store the residue of all the data
        for i in range(0,len(current_input),batch_size) : 
            #since batch size is one, we gave an extra dimension
            data = torch.tensor(current_input[i:i + batch_size])
        
            results = model(data.to(device))[0] # since the [1] is energy prediction
            
            if i == 0 :
                temp = results;
            else  : 
                temp = torch.cat((temp, results))
                
            
        current_false_state = torch.tensor(current_input[:,:,-1]).to(device)
        corrected_state = (current_false_state + temp).cpu().detach().numpy()
        #append this corrected state to the history list
        
        history_list = np.concatenate((history_list, np.expand_dims(corrected_state, axis = 2)), axis = 2)

        print("\nhistory list shape now : " ,history_list.shape)        

        simulation = Simulation_PQP(N = N, DIM =1, BoxSize =1.0, deltat = 0.1, Nsteps = 1, DumpFreq = 100)
        simulation.set_mass(1)
        simulation.set_scaling(True)
        simulation.initialize_pos(scale = 2,file =False)
        simulation.initialize_vel(scale = 1, v_scaling = True)
        
        #initialized the x and v
        for i in range(len(simulation.pos)):
            #initialize the slow position
            simulation.pos[i] = corrected_state[i][0]
            simulation.vel[i] = corrected_state[i][1]
        
        simulation.simulate(track = False)
    
        
        predicted_wrong_pos = simulation.pos
        predicted_wrong_vel = simulation.vel
        
        wrong_prediction = np.concatenate((predicted_wrong_pos,predicted_wrong_vel), axis = 1)
        
        #manipulate the current input, remove the last wrong state

        current_input = np.concatenate((current_input[:,:,1:], np.expand_dims(corrected_state, axis = 2)), axis = 2)
        
    return history_list
           
def create_reference_list(dataloader):
    # the initial list is just in form of n x 2 x 11 the same as data loader
    
    reference_list = np.array([])
    with torch.no_grad():
        for batch_idx, (data,true_residue, _ ) in enumerate(dataloader):
            if batch_idx == 0 :
                reference_list = data # copy the shape
            else:
                reference_list = np.concatenate((reference_list, data), axis = 0)
                
    return reference_list
    
def create_another_timestep(reference_list, total_time, time_step):
    
    #slice every 0.5 hence if timestep is 0.1 slice every 5
    #10 / 0.5 = 20, 
    N = len(reference_list)
    simulation = Simulation_PQP(N = N, DIM =1, BoxSize =1.0, deltat = time_step, Nsteps = int( total_time / time_step ), DumpFreq = 1)
    simulation.set_mass(1)
    simulation.set_scaling(True)
    simulation.initialize_pos(scale = 2,file =False)
    simulation.initialize_vel(scale = 1, v_scaling = True)
    
    #initialized the x and v
    for i in range(len(simulation.pos)):
        #initialize the slow position
        simulation.pos[i] = reference_list[i][0][0]
        simulation.vel[i] = reference_list[i][1][0]
        
    simulation.simulate(track = False)
    
    
    x_list = np.array(simulation.xlist_all)[:,::int(0.1/time_step)]
    vx_list = np.array(simulation.vxlist_all)[:,::int(0.1/time_step)]
    
    data = np.concatenate((np.expand_dims(x_list, axis = 1), np.expand_dims(vx_list, axis = 1)), axis = 1)
    
    return data

if __name__ == "__main__":
    '''using the validation data only since it is considered 
    unseen by the model and compare it to the accurate prediction'''
    
    x_reference = np.load(currdir + '/compiled/accurate_pos_track01.npy')
    vx_reference = np.load(currdir + '/compiled/accurate_vel_track01.npy')
    
    reference = np.concatenate((np.expand_dims(x_reference, axis = 1), np.expand_dims(vx_reference, axis = 1)), axis = 1)
  
    kwargs = {'pin_memory' : True, 'num_workers' : 6, 'shuffle' : False}
    dataset_validation = create_dataset(mode = 'validation', total_time = total_time)
    valid_loader = DataLoader(dataset_validation, batch_size = batch_size , **kwargs)
    reference_list = create_reference_list(valid_loader)
    history_list = process(reference_list, 49.1) # used to be 45.5
    
 # [:,:,10:]
    correlation = np.abs(history_list[:,:,10:] - reference[:,:,10:])
    correlation_avg = np.average(correlation, axis = 0 )
    
    x_axis = [i for i in np.arange(1.0, 50.0 + 0.1, 0.1)]
    x_correlation = correlation_avg[0]
    vx_correlation = correlation_avg[1]
    
    #remember that the first half is generated using 0.05 and truncated every 10 times
    plt.plot(x_axis,x_correlation, label = "q t0.001/ t0.1 + ML")
    
# =============================================================================
    data01 = create_another_timestep(reference, total_time = 50, time_step = 0.1)
    correlation01 = np.abs(reference[:,:,10:] - data01[:,:,10:])
    correlation01_avg = np.average(correlation01, axis = 0 )
    
    x01_correlation = correlation01_avg[0]
    vx01_correlation = correlation01_avg[0]
    plt.plot(x_axis, x01_correlation, label = 'q t0.001/t0.1')
    
# =============================================================================
    data001 = create_another_timestep(reference, total_time = 50, time_step = 0.01) 
    
    correlation001 = np.abs(reference[:,:,10:] - data001[:,:,10:])
    correlation001_avg = np.average(correlation001, axis = 0 )
    
    x001_correlation = correlation001_avg[0]
    vx001_correlation = correlation001_avg[0]
    plt.plot(x_axis, x001_correlation, label = 'q t0.001/t0.01')
# =============================================================================
    
    # plt.plot(x_axis,vx_correlation, label = "p t0.001/ t0.5 + ML")
    
    plt.xticks([i for i in np.arange(1.0, 50.0 + 0.5, 1.0)])
    plt.grid(True)
    plt.legend(loc = "best")
    plt.show()
    
# =============================================================================

    plt.plot(x_axis,vx_correlation, label = "p t0.001/ t0.1 + ML")
    plt.plot(x_axis,vx01_correlation, label = 'p t0.001/t0.1')
    plt.plot(x_axis, vx001_correlation, label = 'p t0.001/t0.01')
    plt.grid(True)
    plt.legend(loc = 'best')
    plt.xticks([i for i in np.arange(1.0, 50.0 + 0.5, 1.0)])
    plt.show()
# =============================================================================

    energy_correlation_reference = np.abs(energy(history_list[:,:,10:]) - energy(reference[:,:,10:]))
    energy_correlation_ML = np.abs(energy(reference[:,:,10:]) - energy(history_list[:,:,10:]))
    energy_correlation_01 = np.abs(energy(reference[:,:,10:]) - energy(data01[:,:,10:]))
    energy_correlation_001 = np.abs(energy(reference[:,:,10:]) - energy(data001[:,:,10:]))
    
    # plt.plot(x_axis,energy_correlation_reference , label = 'E t0.001/t0.001')
    plt.plot(x_axis, energy_correlation_ML,label ='E t0.1 + ML/ t0.001')
    plt.plot(x_axis, energy_correlation_01 ,label = 'E t0.1/t0.001')
    plt.plot(x_axis, energy_correlation_001,label ='E t0.01/ t0.001')
    plt.xticks([i for i in np.arange(1.0, 50.0 + 0.5, 1.0)])
    plt.grid(True)
    plt.legend(loc='best')
    plt.show()

    

    