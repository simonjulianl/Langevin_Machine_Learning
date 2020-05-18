#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 17:35:34 2020

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
import numpy as np
from error import *
from models import *
from collections import defaultdict
from itertools import product
import random
from simulator import simulator_numerical
import argparse
from MD_ML import velocity_verlet_ML

parser = argparse.ArgumentParser('MD Hamiltonian')
args = parser.add_argument('--deltat', type = float, default = 0.01 ,help = "time step")
args = parser.add_argument('--seed' , type = int, default = 1 , help = 'reproducibiltiy')
args = parser.add_argument('--Nsteps' , type = int, default = 150, help = "total number of steps")
args = parser.add_argument('--DumpFreq', type = int, default = 50, help = "slice every dump freq steps")
args = parser.parse_args()
#from MD ML, we have 0.1 * 50 = 5.0 seconds time step 
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
Nsteps = args.Nsteps
DumpFreq = args.DumpFreq
time_step = args.deltat
large_time_step = DumpFreq * time_step

simulator = simulator_numerical(time_step = time_step, Nsteps = Nsteps, DumpFreq = DumpFreq)
train_data, validation_data = simulator.integrate()

model = MLP_General_Hamil(2,10).to(device)
model.load_state_dict(torch.load('ML_General_Hamiltonian.pth'))
model.eval()

def energy(q,p) :
    potential_energy = (q[:,:] ** 2.0 - 1) ** 2.0 + q
    kinetic_energy = (p[:,:] ** 2.0 / 2)
    total_energy = potential_energy + kinetic_energy
    total_energy = np.mean(total_energy, axis = 0)

    return total_energy
    
q_data = np.concatenate((train_data[0],validation_data[0]))
p_data = np.concatenate((train_data[1],validation_data[1]))

q_ML = torch.tensor(np.expand_dims(q_data[:,0], axis=1), dtype = torch.float32).to(device)
p_ML = torch.tensor(np.expand_dims(p_data[:,0], axis=1), dtype = torch.float32).to(device)
    
q_list = np.array(q_ML.cpu()) # this is the list of position
p_list = np.array(p_ML.cpu()) # this is the list of momentum

for t in trange(int(Nsteps/DumpFreq)):
    
    q_ML.requires_grad_(True)
    p_ML.requires_grad_(True)

    temp_q = None
    temp_p = None
    
    for i in range(0,len(q_data), 250): 
        q_next, p_next = velocity_verlet_ML(q_ML[i : (i+250)],p_ML[i : (i+250)], model, large_time_step)
        if i == 0 :
            temp_q = q_next.detach().cpu().numpy()
            temp_p = p_next.detach().cpu().numpy()
        else : 
            temp_q = np.concatenate((temp_q,q_next.detach().cpu().numpy()))
            temp_p = np.concatenate((temp_p, p_next.detach().cpu().numpy()))
    
    q_list = np.concatenate((q_list, temp_q), axis =1)
    p_list = np.concatenate((p_list, temp_p), axis = 1)
    
    q_ML = torch.tensor(temp_q, dtype = torch.float32).to(device)
    p_ML = torch.tensor(temp_p, dtype = torch.float32).to(device)
    print(q_list.shape)
    
#cast to numpy
q_list = q_list
p_list = p_list
ML_energy = energy(q_list, p_list)
reference_energy = energy(q_data,p_data)

assert len(ML_energy) == len(reference_energy)

correlation_energy = np.abs(ML_energy - reference_energy)
correlation_q = np.mean(np.abs(q_list-q_data), axis = 0)
correlation_p = np.mean(np.abs(p_list - p_data), axis = 0)

# =============================================================================
simulator.change(time_step = 0.5, Nsteps = 3, DumpFreq = 1)
train_data_05, validation_data_05 = simulator.integrate()

q_data_05 = np.concatenate((train_data_05[0],validation_data_05[0]))
p_data_05 = np.concatenate((train_data_05[1],validation_data_05[1]))

energy_05 = energy(q_data_05, p_data_05)
correlation_energy_05 = np.abs(energy_05 - reference_energy)
correlation_q_05 = np.mean(np.abs(q_data_05 - q_data), axis = 0)
correlation_p_05 = np.mean(np.abs(p_data_05 - p_data), axis = 0)

# =============================================================================

#ML predictor here used to predict time step of 0.5
x_axis = [i for i in np.arange(0.0, time_step * (Nsteps+1), time_step * DumpFreq)]
xticks = [i for i in np.arange(0.0, time_step * (Nsteps+1), time_step * DumpFreq * 4)]
plt.plot(x_axis, correlation_energy, label = 'energy correlation t0.01/ML predictor 0.5') 
plt.plot(x_axis, correlation_energy_05, label = ' t0.01/ t0.5')
plt.xticks(xticks)
plt.legend(loc = 'best')
plt.grid(True)
plt.show()

plt.plot(x_axis, correlation_q, label = 'q correlation t0.01/ML precictor 0.5')
plt.plot(x_axis, correlation_q_05, label = ' t0.01/ t0.5')
plt.xticks(xticks)
plt.legend(loc = 'best')
plt.grid(True)
plt.show()

plt.plot(x_axis, correlation_p, label = 'p correlation t0.01/ML precictor 0.5')
plt.plot(x_axis, correlation_p_05, label = ' t0.01/ t0.5')
plt.xticks(xticks)
plt.legend(loc = 'best')
plt.grid(True)
plt.show()