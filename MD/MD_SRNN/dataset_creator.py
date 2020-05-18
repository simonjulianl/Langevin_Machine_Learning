#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 14:52:59 2020

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

class MD_Hamiltonian(Dataset):
    def __init__(self,data):
        self.dataset = []
        train_data = data[0]
        train_label = data[1]
        N = len(train_data[0])
        
        for i in range(N):
            state = np.array([train_data[0][i], train_data[1][i]])
            label = np.array([train_label[0][i], train_label[1][i]])
            self.dataset.append((state,label))
        
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return (self.dataset[idx][0], self.dataset[idx][1])
