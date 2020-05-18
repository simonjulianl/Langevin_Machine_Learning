#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 15:48:13 2020

@author: simon
"""

from torch.utils.data import Dataset
import os
import pandas as pd
import transform_data
import numpy as np
import torch
import glob

#data = pd.read_csv('test.csv',delimiter = ',')
#print(data.head()) # this prints the first 5 lines
#
#print(os.getcwd())
#print(os.listdir(os.getcwd()))

#sorted according to index starting from 0 
truth_file = sorted(glob.glob('./configuration_data/output_truth*.txt'))
train_file = sorted(glob.glob('./configuration_data/output_data*.txt'))

class MDdata_loader(Dataset):
    def __init__(self,N,DIM,BoxSize):
        self.sample = []
        for i in range(len(truth_file)):
            #check for all true data first
            filename = truth_file[i]
            true_data = transform_data.read_data(filename)
            
            filename = train_file[i]
            train_data = transform_data.read_data(filename)
            
            true_data = torch.FloatTensor(true_data).resize_(1,DIM,2*N)
            train_data = torch.FloatTensor(train_data).resize_(1,DIM,2*N)
            
        
            stacked = torch.stack((train_data,true_data))

            self.sample.append(stacked)
            
    def __len__(self):
        return len(self.sample)
    
    def __getitem__(self,idx):
        return self.sample[idx][0], self.sample[idx][1]
    
