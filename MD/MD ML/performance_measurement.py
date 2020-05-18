#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 10:24:25 2020

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
from Dataset_creator import load_raw,simulate
import argparse


parser = argparse.ArgumentParser('Fully Connected MD')
args = parser.add_argument('--log_interval',type = int, default = 2000, help = 'log interval to check error')
args = parser.add_argument('--total_time',type = float, default = 100, help ='total N slow step')
args = parser.add_argument('--deltat', type = float, default = 0.05, help = 'time step')
args = parser.add_argument('--epochs', type = int, default = 100, help = 'Number of epochs')
args = parser.add_argument('--seed', type = int, default = 2 , help = 'reproducibiltiy')
args = parser.add_argument('--alpha', type = float, default = 0, help = 'penalty constraint weight, 0 means full residue loss')
args = parser.parse_args()

device = 'cpu'
batch_size = 1
deltat = args.deltat
total_time = int(args.total_time)
alpha = args.alpha

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class MD_Dataset_Sequential(Dataset):
    '''Dataset must be arrange in the form of : 
        1 ) Sequence of pos + vel
        2 ) Seqeunece of pos and vel labels
    '''
    def __init__(self,data):
        self.dataset = []
        train_data = data[0].clone().detach().type(torch.float32)
        label_data = data[1].clone().detach().type(torch.float32)
        final_truth_data = data[2].clone().detach().type(torch.float32)
        
        assert len(train_data) == len(label_data)
     
        for i in range(len(train_data)):
            self.dataset.append((train_data[i],label_data[i], final_truth_data[i])) 
        
                
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,idx):
        return self.dataset[idx][0], self.dataset[idx][1], self.dataset[idx][2]
            
def create_dataset(mode = 'train',total_time_slow = 100,slow_time_step = 0.05,time_difference = 10):
    # =============================================================================
    # The simulation uses asymmetrical double well external force and no interaction between particles    
    # =============================================================================
    '''There are 3 modes : train, validation and test with 60 20 20 % splitting '''
    
    currdir = os.getcwd() + '/compiled/'
    
    if not os.path.exists(currdir + 'true_pos_track_{}3.npy'.format(mode)) : 
        simulate(total_time_slow - time_difference, slow_time_step, jobid = 3)
    true_pos_track = torch.tensor(np.load(currdir + 'true_pos_track_{}3.npy'.format(mode)))
    true_vel_track = torch.tensor(np.load(currdir + 'true_vel_track_{}3.npy'.format(mode)))
    false_last_pos = torch.tensor(np.load(currdir + 'false_last_pos_{}3.npy'.format(mode)))
    false_last_vel = torch.tensor(np.load(currdir + 'false_last_vel_{}3.npy'.format(mode)))
    
    final_data = torch.cat((true_pos_track.unsqueeze(2), true_vel_track.unsqueeze(2)), dim = 2)
    #data is arranged in q,p manner
    label_data = torch.cat((false_last_pos.unsqueeze(1), false_last_vel.unsqueeze(1)), dim = 1)
    
    final_data_processed = final_data[:,::10,:]

    residue_label = final_data[:,-1] - label_data
    final_data_truth = final_data[:,-1]
    

    final_data_processed = torch.cat((final_data_processed[:,:-1],label_data.unsqueeze(1)), dim = 1)
    data = (final_data_processed, residue_label, final_data_truth)
    #all data is processed in q,p manner
    
    dataset = MD_Dataset_Sequential(data)
    
    return dataset

def energy(state):
    return (state[:,0] ** 2 -1) ** 2 + state[:,0] + ( 0.5 * state[:,1] ** 2)

def findStats(train_loader,test_loader, valid_loader):
    # absolute error, hence the data used is |p|, |q| and |E|
    p, q, E = 0, 0, 0
    p_2, q_2, E_2 = 0 ,0 ,0
    
    loaders = [train_loader, test_loader, valid_loader]
    for loader in loaders : 
        for data,label, true_state in loader:
            dq,dp = torch.sum(torch.abs(label),dim = 0)
            p += dp
            p_2 += dp ** 2.0
            
            q += dq
            q_2 += dq ** 2.0
            
            dE = torch.abs(energy(true_state) - energy(data[:,-1])) 
            E += dE
            E_2 += dE ** 2.0

    #make it to expectation
    length = len(train_loader) + len(test_loader) + len(valid_loader)
    p /= length
    p_2 /= length
    q /= length
    q_2 /= length
    E = E.item()/length
    E_2 = E_2.item()/length
    
    print(E, E_2)
    print('q : {:.5f} {:.5f}'.format(q, (q_2 - q ** 2.0)) )
    print('p : {:.5f} {:.5f}'.format(p, (p_2 - p ** 2.0)) )
    print('E : {:.5f} {:.5f}'.format(E, (E_2 - E ** 2.0)) )
        
if __name__ == "__main__":
    #total time slow is the number of time step performed using 0.05
    #the difference is in the last time step
    dataset_train = create_dataset(mode = 'train', total_time_slow = total_time, slow_time_step = deltat, time_difference = 2)
    # dataset_train.dataset = dataset_train.dataset[:10]
    kwargs = {'pin_memory' : True, 'num_workers' : 6, 'shuffle' : True}
    train_loader = DataLoader(dataset_train, batch_size = batch_size, **kwargs)
    
    dataset_validation = create_dataset(mode = 'validation', total_time_slow = total_time, slow_time_step = deltat, time_difference = 10)
    valid_loader = DataLoader(dataset_validation, batch_size = batch_size , **kwargs)
    
    dataset_test = create_dataset(mode = 'test', total_time_slow = total_time, slow_time_step = deltat, time_difference = 2)
    test_loader = DataLoader(dataset_test, batch_size = batch_size , **kwargs)
    
    print(next(iter(train_loader)))
    data,label, true_state = next(iter(test_loader))
    
    print(data.transpose(1,2).shape)
    findStats(train_loader, test_loader, valid_loader)
    
