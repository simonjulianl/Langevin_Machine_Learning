#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 10:12:50 2020

@author: simon
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset,DataLoader
from Dataset_creator import create_dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm,trange

import os
import torchvision.models as models
from torchsummary import summary
import argparse

parser = argparse.ArgumentParser('Fully Connected MD')
args = parser.add_argument('--log_interval',type = int, default = 2000, help = 'log interval to check error')
args = parser.add_argument('--total_time',type = float, default = 10, help ='total N slow step')
args = parser.add_argument('--deltat', type = float, default = 0.05, help = 'time step')
args = parser.add_argument('--epochs', type = int, default = 20, help = 'Number of epochs')
args = parser.add_argument('--seed', type = int, default = 0 , help = 'reproducibiltiy')
args = parser.add_argument('--alpha', type = float,default = 0 ,help = 'weightage between energy and pos loss')
args = parser.parse_args()

# =============================================================================
# Hyperparameter setting
# =============================================================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 128
deltat = args.deltat
total_time = int(args.total_time)
alpha = args.alpha #0 means full pos/vel loss and 1 means full energy loss

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class MDModel(nn.Module):
    def __init__(self):
        super(MDModel,self).__init__()
        
        self.conv1d = nn.Sequential(
            nn.Conv1d(2,40, kernel_size = 1,stride = 1),
            nn.Conv1d(40,80, kernel_size = 1, stride = 1),
            nn.Conv1d(80,160, kernel_size = 2 , stride = 1),
            )
        
        self.conv2d = nn.Sequential(
            nn.Conv2d(1,10,kernel_size = 1,stride =1),
            nn.Conv2d(10,10,kernel_size = 1, stride = 1),
            nn.Conv2d(10,1,kernel_size = 2,stride = 1),
            )
        
        self.fc1 = nn.Sequential(
                nn.Linear(4,160),
                nn.Softplus(),
                # nn.BatchNorm1d(160),
                nn.Linear(160,80),
                nn.Softplus(),
                nn.Linear(80,40),
                nn.Softplus(),
                nn.Linear(40,2),
            )
        
    def forward(self,x):
        # x_pos = self.conv1d(x.squeeze())
        # x_pos = x.squeeze()
        x_pos = x
        x_pos = x_pos.view(-1,np.prod(list(x_pos.shape[1:])))
        
        energy = self.conv2d(x)
        return self.fc1(x_pos),energy
        
def loss_function_pos_vel(initial_state,final_state):
    # criterion = nn.MSELoss(reduction = 'sum')
    criterion = nn.SmoothL1Loss(reduction = 'sum')
    return criterion(initial_state,final_state)
        
def loss_function_energy(initial_energy,final_state):
    # criterion = nn.MSELoss(reduction = 'mean')
    criterion = nn.SmoothL1Loss(reduction = 'sum')
    final_energy = (final_state[:,0] ** 2 - 1) ** 2 + final_state[:,0] + ( 0.5 * final_state[:,1] ** 2)
    return criterion(initial_energy.squeeze(),final_energy)

def combined_loss(initial_energy, initial_state, final_state):
    return alpha * loss_function_energy(initial_energy, final_state) + (1 - alpha) * loss_function_pos_vel(initial_state, final_state)
    
model = MDModel().to(device)
# summary(model,input_size = (1,2,2))
optimizer = optim.AdamW(model.parameters(), lr = 1e-4, weight_decay = 0.1) #adam with weight decay
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-3, max_lr=1e-2,cycle_momentum = False)

dataset_train = create_dataset(mode = 'train', total_time_slow = total_time, slow_time_step = deltat, plot = False)
# dataset_train.dataset = dataset_train.dataset[:10]
kwargs = {'pin_memory' : True, 'num_workers' : 6, 'shuffle' : True}
train_loader = DataLoader(dataset_train, batch_size = batch_size, **kwargs)

dataset_validation = create_dataset(mode = 'validation', total_time_slow = total_time, slow_time_step = deltat, plot = False)
valid_loader = DataLoader(dataset_validation, batch_size = batch_size , **kwargs)

dataset_test = create_dataset(mode = 'test', total_time_slow = total_time, slow_time_step = deltat, plot = False)
test_loader = DataLoader(dataset_test, batch_size = batch_size , **kwargs)

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, true_state) in enumerate(train_loader): 
        # =====================================================================
        # here the data is in the form of initial state + large final step state
        # true_state is the true final state with smaller small step 
        # =====================================================================
        data = data.unsqueeze(1).to(device)
        true_state = true_state.to(device)
        optimizer.zero_grad()
        predicted_state, predicted_energy = model(data)
        loss = combined_loss(predicted_energy, predicted_state, true_state)
        loss.backward()
        
        train_loss += loss
        
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch : {} [{}/{} ({:.2f} %)] \t Loss : {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss / len(data),
                ))
        scheduler.step()
    print('====> Epoch : {} Average Loss : {:.10f}'.format(
        epoch, train_loss / len(train_loader.dataset)
        ))
    train_loss /= len(train_loader.dataset)
  
    return train_loss
    
def valid(epoch):
    model.eval()
    test_loss = 0
    pos_loss = 0
    with torch.no_grad():
        for batch_idx, (data,true_state) in enumerate(valid_loader):
            data = data.unsqueeze(1).to(device)
            true_state = true_state.to(device)
            predicted_state, predicted_energy = model(data)
            loss = combined_loss(predicted_energy, predicted_state, true_state)
            test_loss += loss
            
            loss_pos = loss_function_pos_vel(predicted_state, true_state)
            pos_loss += loss_pos
            if batch_idx % args.log_interval == 0:
                print('Test Epoch : {} [{}/{} ({:.2f} %)] \t Loss : {:.6f}'.format(
                    epoch, batch_idx * len(data), len(test_loader.dataset),
                    100. * batch_idx / len(test_loader),
                    loss / len(data),
                    ))
            
    test_loss /= len(test_loader.dataset)
    pos_loss /= len(test_loader.dataset)
    print('====> Validation Test Loss : {:.6f}'.format(test_loss))
    print('====> Validation Position / Velocity Test Loss : {:.10f}'.format(pos_loss))

    return test_loss
    
def test():
    
    criterion = nn.MSELoss(reduction = 'sum')
    epoch_loss = 0
    
    model.eval()
    l1error_position = 0
    l1error_position_2 = 0 # expectation of delta q ^ 2
    
    l1error_velocity = 0
    l1error_velocity_2 = 0 # expectation of delta p ^ 2
    with torch.no_grad():
        for batch_idx, (data, true_state) in enumerate(test_loader):
            data = data.unsqueeze(dim = 1).to(device)
            true_state = true_state.to(device)
            predicted_state, predicted_energy = model(data)
            data = data.squeeze().detach().cpu().numpy()
            predicted_state = predicted_state.detach().cpu().numpy()
            true_state = true_state.detach().cpu().numpy()
                    
            position_diff = np.abs(predicted_state[:,0] - true_state[:,0])
            velocity_diff = np.abs(predicted_state[:,1] - true_state[:,1])
            
            l1error_position += np.sum(position_diff)
            l1error_velocity += np.sum(velocity_diff)
            
            l1error_position_2 += np.sum(position_diff ** 2.0)
            l1error_velocity_2 += np.sum(velocity_diff ** 2.0)
            
            loss = criterion(torch.tensor(predicted_state),torch.tensor(true_state))
            epoch_loss += loss.item()
            
            print('initial state : {} , predicted state : {}, true state : {}'.format(
                data[0],predicted_state[0],true_state[0]))
    
    pos_diff_expectation = l1error_position / len(test_loader.dataset)
    vel_diff_expectation = l1error_velocity / len(test_loader.dataset)
    pos2_diff_expectation = l1error_position_2 / len(test_loader.dataset)
    vel2_diff_expectation = l1error_velocity_2 / len(test_loader.dataset)
    
    print('L1 Average error position : {:.4f}'.format(pos_diff_expectation)) 
    print('position error variance : {:.4f}'.format(pos2_diff_expectation - pos_diff_expectation ** 2.0)) 
    
    print('L1 Average error velocity : {:.4f}'.format(vel_diff_expectation))
    print('velocity error variance : {:.4f}'.format(vel2_diff_expectation - vel_diff_expectation ** 2.0)) 
    
    epoch_loss /= len(test_loader.dataset)
    
    return epoch_loss 
    model.eval()

            
def original_error():
    original_loss = 0
    with torch.no_grad():
        for batch_idx, (data,true_state) in enumerate(train_loader):
            data = data.unsqueeze(1).to(device)
            true_state = true_state.to(device)
            data = data.squeeze()
            loss = loss_function_pos_vel(data[:,1], true_state)
            original_loss += loss.item()
            
    original_loss /= len(train_loader.dataset)
    return original_loss 
    
if __name__ == '__main__':
    print('MD Machine Learning Fully Connected')
    # print('Original position error: {:.6f} '.format(original_error()))
    train_error_list = []
    test_error_list = []
    for epoch in range(1, args.epochs + 1):
        train_error_list.append(train(epoch))
        test_error_list.append(valid(epoch))
    print('L2 Loss pos/vel : {:.4f} '.format(test()))
    
    fig = plt.figure(figsize = (10,10))
    plt.plot(train_error_list, label = 'train loss')
    plt.plot(test_error_list, label = 'validation loss')
    plt.legend(loc = 'best')
    torch.save(model.state_dict(),'FC1')

