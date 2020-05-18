#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 08:19:30 2020

@author: simon
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset,DataLoader
from MS_Data_Creator import create_dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm,trange

import os
import argparse

parser = argparse.ArgumentParser('Fully Connected MD')
args = parser.add_argument('--log_interval',type = int, default = 2000, help = 'log interval to check error')
args = parser.add_argument('--total_time',type = float, default = 10, help ='total N slow step')
args = parser.add_argument('--deltat', type = float, default = 0.05, help = 'time step')
args = parser.add_argument('--epochs', type = int, default = 100, help = 'Number of epochs')
args = parser.add_argument('--seed', type = int, default = 2 , help = 'reproducibiltiy')
args = parser.add_argument('--alpha', type = float, default = 0, help = 'penalty constraint weight, 0 means full residue loss')
args = parser.parse_args()

device = 'cpu'
batch_size = 2
deltat = args.deltat
total_time = int(args.total_time)
alpha = args.alpha

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class energyLayer(nn.Module):  
    def __init__(self):
        super(energyLayer,self).__init__()
        
    def forward(self, state):
        # return torch.mean((state[:,0] ** 2 -1) ** 2 + state[:,0] + ( 0.5 * state[:,1] ** 2),dim = 1)
        return (state[:,0] ** 2 -1) ** 2 + state[:,0] + ( 0.5 * state[:,1] ** 2)

class pqDifference(nn.Module):
    def __init__(self, false_state):
        super(pqDifference,self).__init__()
        self.state = false_state
        
    def forward(self,state):
        state = state - self.state
        return state 
        
#dummy model for testing
class model(nn.Module):
    def __init__(self):
        super(model,self).__init__()
        self.layer = nn.Sequential(
            energyLayer()
            )
        
    def forward(self,x, true_state): #take average of the first 10 states as the reference energy
        # print(true_state.shape)
        # self.layer2 = nn.Sequential(
        #     pqDifference(true_state)
        #     )
        # test = torch.FloatTensor([[1,1],[2,2]])
        # print(self.layer2(test))
        return self.layer(x)
        
if __name__ == "__main__":
    model = model()
    dataset_train = create_dataset(mode = 'train', total_time_slow = total_time, slow_time_step = deltat, time_difference = 10)
    # dataset_train.dataset = dataset_train.dataset[:10]
    kwargs = {'pin_memory' : True, 'num_workers' : 6, 'shuffle' : False}
    train_loader = DataLoader(dataset_train, batch_size = batch_size, **kwargs)
    
    dataset_validation = create_dataset(mode = 'validation', total_time_slow = total_time, slow_time_step = deltat, time_difference = 10)
    valid_loader = DataLoader(dataset_validation, batch_size = batch_size , **kwargs)
    
    dataset_test = create_dataset(mode = 'test', total_time_slow = total_time, slow_time_step = deltat, time_difference = 10)
    test_loader = DataLoader(dataset_test, batch_size = batch_size , **kwargs)
    
    print(next(iter(test_loader)))
    data,label, true_state = next(iter(test_loader))
    
    print(data.transpose(1,2).shape)
    print(model(data.transpose(1,2), true_state))

