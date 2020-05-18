#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 14:07:05 2020

@author: simon
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from LSTM_Data_Creator import create_dataset
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
from tqdm import trange
import numpy as np

parser = argparse.ArgumentParser('MD Multiple States FC Concurrent predictor')
args = parser.add_argument('--log_interval', type = int, default = 1000, help = 'log interval for loss')
args = parser.add_argument('--total_time', type = float ,default = 10, help = 'Total N steps', metavar = 'N')
args = parser.add_argument('--deltat', type = float ,default = 0.05, help = 'time step')
args = parser.add_argument('--epochs', type = int, default = 100 , help = 'Total Epochs')
args = parser.add_argument('--seed', type = int , default = 1 , help = 'seed for reproducibility')
args = parser.add_argument('--alpha', type = float ,default = 0.5 ,help = 'balance between pos and energy loss, 0 means full position loss')
args = parser.parse_args()
# =============================================================================
# Hyper parameter Setting
# =============================================================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 128
deltat = args.deltat
total_time = int(args.total_time)
alpha = args.alpha
INITIAL_DIM = int(args.total_time) #all the history in the p and q positions are used as input
FINAL_DIM = INITIAL_DIM #Since it is a 1-1 correspondence

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

dataset_train = create_dataset(mode = 'train' , total_time_slow = total_time, slow_time_step = deltat)
kwargs = {'pin_memory' : True, 'num_workers' : 6, 'shuffle' : True}
train_loader = DataLoader(dataset_train, batch_size = batch_size, **kwargs)

dataset_validation = create_dataset(mode = 'validation', total_time_slow = total_time, slow_time_step = deltat)
valid_loader = DataLoader(dataset_validation, batch_size = batch_size , **kwargs)

dataset_test = create_dataset(mode = 'test', total_time_slow = total_time, slow_time_step = deltat)
test_loader = DataLoader(dataset_test, batch_size = batch_size , **kwargs)


class MDModel(nn.Module):
    def __init__(self):
        super(MDModel,self).__init__()
        
        self.conv1d = nn.Sequential(
            nn.Conv1d(INITIAL_DIM,40, kernel_size = 1,stride = 1),
            nn.Conv1d(40,80, kernel_size = 1, stride = 1),
            nn.Conv1d(80,160, kernel_size = 2 , stride = 1),
            )
        
          
        self.fc1 = nn.Sequential( 
                # nn.Linear(INITIAL_DIM * 2,160),
                nn.Linear(160,160), #This is for convolution
                nn.Softplus(),
                # nn.BatchNorm1d(160),
                nn.Linear(160,80),
                nn.Softplus(),
                nn.Linear(80,40),
                nn.Softplus(),
                nn.Linear(40,FINAL_DIM * 2),
            )
        
    def forward(self,x):
        x_pos = self.conv1d(x.squeeze())
        # x_pos = x.squeeze()
        x_pos = x_pos.view(-1,np.prod(list(x_pos.shape[1:])))
        
        return self.fc1(x_pos)
    
model = MDModel().to(device)
optimizer = optim.Adam(model.parameters() , lr = 1e-3, amsgrad = True)
# optimizer = optim.AdamW(model.parameters() ,lr = 1e-3, weight_decay = 1e-2)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr = 1e-4, max_lr = 1e-3, cycle_momentum = False)
#testing with random data 
with torch.no_grad():
    data,label = next(iter(train_loader))
    output = model(data.to(device))

    
def posvel_loss(predicted_state,true_state):
    criterion = nn.MSELoss(reduction = 'sum')
    # criterion = nn.SmoothL1Loss(reduction = 'sum')
    return criterion(predicted_state,true_state)
   
def loss_function_energy(initial_state,final_state):
    criterion = nn.MSELoss(reduction = 'sum')
    # criterion = nn.SmoothL1Loss(reduction = 'sum')
   
    initial_energy = (initial_state[:,:,0] ** 2 -1) ** 2 + initial_state[:,:,0] + ( 0.5 * initial_state[:,:,1] ** 2)
    final_energy = (final_state[:,:,0] ** 2 - 1) ** 2 + final_state[:,:,0] + ( 0.5 * final_state[:,:,1] ** 2)
    return criterion(initial_energy,final_energy)

def combined_loss(true_state, predicted_state, initial_state):
    return alpha * loss_function_energy(initial_state, predicted_state) + (1-alpha) * posvel_loss(true_state, predicted_state)
 
training_loss = []
validation_loss = []

def train(epoch):
    epoch_loss = 0
    for batch_idx, (data, label_sequence) in enumerate(train_loader):
        optimizer.zero_grad()
        data,true_state = data.to(device), label_sequence.to(device)
        predicted_state = model(data).view(-1, FINAL_DIM,2)
        initial_state = data
        
        loss = combined_loss(true_state, predicted_state, initial_state)
        loss.backward()
        
        optimizer.step()
        epoch_loss += loss.item()
        
        # scheduler.step()
        if batch_idx % args.log_interval == 0:
                print('Train Epoch : {} [{}/{} ({:.2f} %)] \t Loss : {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss / len(data),
                    ))
       
    
    epoch_loss /= len(train_loader.dataset)
    
    return epoch_loss

def validation(epoch):
    epoch_loss = 0
    for batch_idx, (data, label_sequence) in enumerate(valid_loader):
        data,true_state = data.to(device), label_sequence.to(device)
        predicted_state = model(data).view(-1, FINAL_DIM ,2)
        initial_state = data
 
        loss = combined_loss(true_state, predicted_state, initial_state)
        
        epoch_loss += loss.item()
        
        if batch_idx % args.log_interval == 0:
            print('Validation Epoch : {} [{}/{} ({:.2f} %)] \t Loss : {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss / len(data),
                ))
                
    epoch_loss /= len(valid_loader.dataset)
    
    return epoch_loss

def test():
    
    criterion = nn.MSELoss(reduction = 'sum')
    epoch_loss = 0
    
    model.eval()
    l1error_position = 0
    l1error_position_2 = 0 # expectation of delta q ^ 2
    
    l1error_velocity = 0
    l1error_velocity_2 = 0 # expectation of delta p ^ 2
    with torch.no_grad():
        for batch_idx, (data, label_sequence) in enumerate(test_loader):
            data, true_state = data.to(device), label_sequence.to(device)
            predicted_state = model(data).view(-1,FINAL_DIM,2)
            
            predicted_state = predicted_state.detach().cpu().numpy()
            true_state = true_state.detach().cpu().numpy()
        
            position_diff = np.abs(predicted_state[:,-1,0] - true_state[:,-1,0])
            velocity_diff = np.abs(predicted_state[:,-1,1] - true_state[:,-1,1])
            
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
    
    print('L1 Average error position : {:.6f}'.format(pos_diff_expectation)) 
    print('position error variance : {:.6f}'.format(pos2_diff_expectation - pos_diff_expectation ** 2.0)) 
    
    print('L1 Average error velocity : {:.6f}'.format(vel_diff_expectation))
    print('velocity error variance : {:.6f}'.format(vel2_diff_expectation - vel_diff_expectation ** 2.0)) 
    
    epoch_loss /= len(test_loader.dataset)
    
    return epoch_loss 

if __name__ == '__main__':
    for i in range(args.epochs):
        training_loss.append(train(i))
        validation_loss.append(validation(i))
        
    print('L2 Loss pos/vel : {:.4f} '.format(test()))
    
    fig = plt.figure(figsize = (10, 10))
    plt.plot(training_loss, label = 'training_loss')
    plt.plot(validation_loss, label = 'validation_loss')
    plt.legend(loc = 'best')

