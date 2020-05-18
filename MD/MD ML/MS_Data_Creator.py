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
            
def create_dataset(mode = 'train',total_time_slow = 5,slow_time_step = 0.05,time_difference = 10):
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
    label_data = torch.cat((false_last_pos.unsqueeze(1), false_last_vel.unsqueeze(1)), dim = 1)
    
    final_data_processed = final_data[:,::10,:]
  
    residue_label = final_data[:,-1] - label_data
    final_data_truth = final_data[:,-1]

# final_data_processed = torch.cat((final_data_processed[:,:-1],label_data.unsqueeze(1)), dim = 1) only use this when the RestNstep = 10
    final_data_processed = torch.cat((final_data_processed[:,:],label_data.unsqueeze(1)), dim = 1)
    data = (final_data_processed, residue_label, final_data_truth)
    dataset = MD_Dataset_Sequential(data)
    
    return dataset

