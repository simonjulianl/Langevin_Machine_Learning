#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 12:35:34 2020

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
from multiprocessing import Pool
import matplotlib.pyplot as plt
import random
from correlator_data_generator import simulate

''' All needed files : 
    correlator_data_generator 
    simulator_pqp
    correlator_dataset
'''

class MD_Dataset_Sequential(Dataset):
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
            
def create_dataset(mode = 'train',total_time = 10):
    # =============================================================================
    # The simulation uses asymmetrical double well external force and no interaction between particles    
    # total time here refers to deltat * Nsteps where deltat = 0.001 as reference fixed
    # and the one being tested using ML is using 0.5 timestep with data generated using 0.05 hence t0 t10 ...
    # =============================================================================
    '''There are 2 modes : train and validation with 60 40% splitting '''
    
    currdir = os.getcwd() + '/compiled/'
    
    #since we want max t = 10, and tau = 0.001 hence we need 10000 steps to achieve that 
    if not os.path.exists(currdir + 'ml_data_{}_01.npy'.format(mode)) : 
        simulate(total_time / 0.001, 0.001)
    
    dataset = torch.tensor(np.load(currdir + 'ml_data_{}_01.npy'.format(mode)))
    true_state = torch.tensor(np.load(currdir + 'ml_truestate_{}_01.npy'.format(mode)))
        
    #create residue by collecting false state
    false_state = dataset[:,:,-1].squeeze()
    residue = true_state - false_state

    data = (dataset, residue, true_state)
    
    #create torch dataloader
    dataloader = MD_Dataset_Sequential(data)  
    
    return dataloader

if __name__ == "__main__":
    data = create_dataset(mode = "train", total_time = 10);
    residue_xlist = []
    residue_vxlist = []
    for _, residue ,_ in data : 
        residue_xlist.append(residue[0].item())
        residue_vxlist.append(residue[1].item())
    residue_xlist = np.array(residue_xlist)
    residue_vxlist = np.array(residue_vxlist)
    plt.hist(residue_vxlist, density = True, bins = 500)
    plt.xlabel('vx residue')
    plt.ylabel('probabiltiy')
    plt.show()
    


