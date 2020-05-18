#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  12 14:30:25 2020

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
import argparse 
import os
from error import *
from models import *
from dataset_creator import MD_Hamiltonian
from simulator import simulator_numerical
from torch.autograd import grad

# 937162211
parser = argparse.ArgumentParser('MD Hamiltonian')
args = parser.add_argument('--log_interval', type= int, default = 1000, help="print message")
args = parser.add_argument('--epochs', type = int , default = 100, help = "Number of epochs")
args = parser.add_argument('--deltat', type = float, default = 0.01 ,help = "time step")
args = parser.add_argument('--seed' , type = int, default =937162211 , help = 'reproducibiltiy')
args = parser.add_argument('--Nsteps' , type = int, default = 50, help = "total number of steps")
args = parser.add_argument('--batch_size' , type = int, default = 32, help = "batch size")
args = parser.add_argument('--lr', type= float , default = 1e-3, help="learning rate")
args = parser.add_argument('--scheduler' ,type = bool ,default = False, help ="reduce of plateau scheduler")
args = parser.parse_args()
# =============================================================================
# Hyper Parameter Tuning  
# =============================================================================
batch_size = args.batch_size
device = "cuda" if torch.cuda.is_available() else "cpu"
time_step = args.deltat
Nsteps = args.Nsteps
lr = args.lr # learning rate
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(args.seed) # apparently this is required
torch.cuda.manual_seed_all(args.seed) # gpu vars
large_time_step = Nsteps * time_step


def data_stats():
    correlator_train = 0
    for (data,label) in train_loader:
        correlator_train += torch.sum(torch.abs(data-label))
    print('initial correlation : ', correlator_train / len(train_loader.dataset))


# Func here meaning the deep learning approximation to find the hamiltonian
# all code is written in torch to allow backprop
# default force here is double well
def derivative_ML(q, p, Func_dqdt, Func_dpdt) -> tuple: # return tuple of next time step
    '''here there is periodic boundary condition since we are using double well
    with potential barrier at x= 2 and x = -2, M and Temperature = 1 by default for simplicity '''
    #here p and q is 1 dimensional    
    # hamiltonian = Func_dpdt(q,p) # we need to sum because grad can only be done to scalar
    # dpdt = -grad(hamiltonian.sum(), q, create_graph = True)[0] # dpdt = -dH/dq
    
    KE = Func_dpdt(p,q) # only kinetic part of the hamiltonian is used here
    dpdt = -grad(KE.sum(), q, create_graph = True)[0] # dpdt = -dH/dq
    #if the create graph is false, the backprop will not update the it and hence we need to retain the graph

    # hamiltonian = Func_dqdt(q,p)
    # dqdt = grad(hamiltonian.sum(), p, create_graph = True)[0] #dqdt = dH/dp
    
    Potential = Func_dqdt(p,q) # only potential part of the hamiltonian is used here 
    dqdt = grad(Potential.sum(), p, create_graph = True)[0] # dpdt = -dH/dq

    return (dqdt, dpdt) # all data is arrange in q p manner
      
def training(epoch) -> float:
    model_dpdt.train()
    model_dqdt.train()
    criterion = nn.MSELoss()
    
    train_loss = 0
    for batch_idx, (data, label) in enumerate(train_loader):
 
        q = torch.tensor(data[:,0], dtype=torch.float32).unsqueeze(1).to(device)
        p = torch.tensor(data[:,1], dtype=torch.float32).unsqueeze(1).to(device)
        
        dqdt_true = torch.tensor(label[:,0], dtype=torch.float32).unsqueeze(1).to(device)
        dpdt_true = torch.tensor(label[:,1], dtype=torch.float32).unsqueeze(1).to(device)
        
        q.requires_grad_(True)
        p.requires_grad_(True)
    
        optim_dpdt.zero_grad()
        optim_dqdt.zero_grad()
        
        dqdt, dpdt = derivative_ML(q, p, model_dqdt, model_dpdt)
        loss = criterion(dqdt, dqdt_true) + criterion(dpdt, dpdt_true)
        loss.backward()
        train_loss += loss
        
        optim_dpdt.step()
        optim_dqdt.step()
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch : {} [{}/{} ({:.2f} %)] \t Loss : {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss / len(data),
                ))
        
        if args.scheduler : 
            scheduler_dqdt.step(loss)
            scheduler_dpdt.step(loss)
            
    print('====> Epoch : {} Average Loss : {:.10f}'.format(
        epoch, train_loss / len(train_loader.dataset)
        ))
    train_loss /= len(train_loader.dataset)
    return train_loss

def validating(epoch) -> float :
    model_dqdt.eval()
    model_dpdt.eval()
    criterion = nn.MSELoss()
    validation_loss = 0
    
    correlator_p, correlator_q = 0,0 # same like the previous correlator
    for batch_idx, (data, label) in enumerate(validation_loader):
        q = torch.tensor(data[:,0], dtype=torch.float32).unsqueeze(1).to(device)
        p = torch.tensor(data[:,1], dtype=torch.float32).unsqueeze(1).to(device)
        
        dqdt_true = torch.tensor(label[:,0], dtype=torch.float32).unsqueeze(1).to(device)
        dpdt_true = torch.tensor(label[:,1], dtype=torch.float32).unsqueeze(1).to(device)
        
        q.requires_grad_(True)
        p.requires_grad_(True)
        
        dqdt, dpdt = derivative_ML(q, p, model_dqdt, model_dpdt)
        loss = criterion(dqdt, dqdt_true) + criterion(dpdt, dpdt_true)
        
        q_next = q + dqdt * large_time_step
        p_next = p + dpdt * large_time_step
        
        q_true = q + dqdt_true * large_time_step
        p_true = p + dpdt_true * large_time_step
        
        correlator_q += torch.sum(torch.abs(q_next - q_true))
        correlator_p += torch.sum(torch.abs(p_next - p_true))
        
        validation_loss += loss
        
        if batch_idx % args.log_interval == 0:
            print('Validation Epoch : {} [{}/{} ({:.2f} %)] \t Loss : {:.6f}'.format(
                epoch, batch_idx * len(data), len(validation_loader.dataset),
                100. * batch_idx / len(validation_loader),
                loss / len(data),
                ))
            
    correlator_p /= len(validation_loader.dataset)
    correlator_q /= len(validation_loader.dataset)
    print('====> Epoch : {} Average Loss : {:.10f} \nCorrelation q  : {} correlation p : {}'.format(
        epoch, validation_loss / len(validation_loader.dataset), correlator_q, correlator_p
        ))
    validation_loss /= len(validation_loader.dataset)
    return validation_loss

if __name__ == "__main__":
    model_dqdt = MLP2H_General_Hamil(2, 10).to(device) # you need KE to get dqdt
    model_dpdt = MLP2H_General_Hamil(2, 10).to(device) # you need potential to get dpdt
    optim_dqdt = torch.optim.Adam(model_dqdt.parameters(), lr = lr)
    optim_dpdt = torch.optim.Adam(model_dpdt.parameters(), lr = lr)
    scheduler_dpdt = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_dpdt, 'min', verbose = False, factor = 0.99) # use default patience of 10, factor = 0.1
    scheduler_dqdt = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_dqdt, 'min', verbose = False, factor = 0.99) 
    
    simulator = simulator_numerical(time_step = time_step, Nsteps = Nsteps, DumpFreq = Nsteps) # we just need to first and last for accurate
    train, validation = simulator.integrate()
    
    train_data = (train[0][:,0], train[1][:,0]) # first index 0 is q and 1 is p
    train_label = ((train[0][:,-1]-train[0][:,0])/large_time_step, (train[1][:,-1]-train[1][:,0])/large_time_step) # train label is delta q , p / delta t instead of next q,p
    
    training_data = (train_data,train_label)
    train_dataset = MD_Hamiltonian(training_data)
    
    kwargs = {'num_workers' : 0, 'pin_memory': True, 'shuffle' : True}
    train_loader = DataLoader(train_dataset, batch_size = batch_size, **kwargs)
    
    validation_data = (validation[0][:,0], validation[1][:,0]) # first index 0 is q and 1 is p
    validation_label = ((validation[0][:,-1] - validation[0][:,0])/large_time_step, (validation[1][:,-1] -  validation[1][:,0])/ large_time_step)
    validating_data = (validation_data, validation_label)
    validation_dataset = MD_Hamiltonian(validating_data)
    
    validation_loader = DataLoader(validation_dataset, batch_size = batch_size, ** kwargs)
    data_stats()
    # testing
#################################################################################
    data,label = next(iter(train_loader))
    q = torch.tensor(data[:,0], dtype=torch.float32).unsqueeze(1).to(device)
    p = torch.tensor(data[:,1], dtype=torch.float32).unsqueeze(1).to(device)
    
    dqdt_true = torch.tensor(label[:,0], dtype=torch.float32).unsqueeze(1).to(device)
    dpdt_true = torch.tensor(label[:,1], dtype=torch.float32).unsqueeze(1).to(device)
    
    
    q.requires_grad_(True)
    p.requires_grad_(True)
    
    dqdt, dpdt = derivative_ML(q, p, model_dqdt, model_dpdt)
    
##################################################################################
    train_loss = []
    validation_loss = []
    for i in range(args.epochs):
        train_loss.append(training(i))
        validation_loss.append(validating(i))
        
    plt.plot(train_loss)
    plt.plot(validation_loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("training curve")
    plt.legend(['train loss', 'validation loss'])
    torch.save({'model_dqdt' : model_dqdt.state_dict(),
                'model_dpdt' : model_dpdt.state_dict()}, 'ML_Hamiltonian05_2models_seed937162211.pth')