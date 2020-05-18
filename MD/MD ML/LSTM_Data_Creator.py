#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 12:03:45 2020

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

class MD_Dataset_Sequential(Dataset):
    '''Dataset must be arrange in the form of : 
        1 ) Sequence of pos + vel
        2 ) Seqeunece of pos and vel labels
    '''
    def __init__(self,data, LSTM):
        self.dataset = []
        train_data = data[0].clone().detach().type(torch.float32)
        label_data = data[1].clone().detach().type(torch.float32)
        
        assert len(train_data) == len(label_data)
     
        for i in range(len(train_data)):
            if LSTM : 
                self.dataset.append((train_data[i,:10:,],label_data[i,:10:])) #LSTM based, input 10 label 10 
            else : 
                # self.dataset.append((train_data[i,:10,],label_data[i,-1])) # given t1,.. t10 predict tlast
                self.dataset.append((train_data[i,:100:10,:], label_data[i,-1,:])) # given t0, t10, t20, ... , predicts t100
                
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,idx):
        return self.dataset[idx][0], self.dataset[idx][1]
            
def create_dataset(mode = 'train',total_time_slow = 5,slow_time_step = 0.05, LSTM = False):
    # =============================================================================
    # The simulation uses asymmetrical double well external force and no interaction between particles    
    # =============================================================================
    '''There are 3 modes : train, validation and test with 60 20 20 % splitting '''
    
    currdir = os.getcwd() + '/compiled/'
    
    if not os.path.exists(currdir + 'true_pos_track_{}2.npy'.format(mode)) : 
        simulate(total_time_slow, slow_time_step, jobid = 2, LSTM = False)
    true_pos_track = torch.tensor(np.load(currdir + 'true_pos_track_{}2.npy'.format(mode)))
    true_vel_track = torch.tensor(np.load(currdir + 'true_vel_track_{}2.npy'.format(mode)))
    last_pos = torch.tensor(np.load(currdir + 'final_pos_track_{}2.npy'.format(mode)))
    last_vel = torch.tensor(np.load(currdir + 'final_vel_track_{}2.npy'.format(mode)))
    
    final_data = torch.cat((true_pos_track.unsqueeze(2), true_vel_track.unsqueeze(2)), dim = 2)
    last_pos = torch.cat((last_pos.unsqueeze(1), last_vel.unsqueeze(1)), dim = 1)
    train_label = torch.cat((final_data[:,1:],last_pos.unsqueeze(1)), dim = 1)
    
    data = (final_data,train_label)
    dataset = MD_Dataset_Sequential(data,LSTM)
    
    return dataset 
