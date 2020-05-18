#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 13:02:43 2020

@author: simon
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 11:02:53 2020

@author: simon
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset,DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm,trange
from custom_layers import pqDifference, energyLayer

import os
import torchvision.models as models
from torchsummary import summary
import argparse
from correlator_dataset import create_dataset
''' Files needed : 
    custom_layers
    correlator_data_generator
    correlator_dataset : create_dataset(mode, total_time)
        1) total time is calculated with respect to tau = 0.001 for reference
            hence total time 0.001 means 
            deltat is fixed at 0.001 as reference and the data is fed in t0 t0.5 t1.0 manner 
            so in total there are
            t0 t0.5 t1.0 t1.5 t2.0 t2.5 t3.0 t3.5 t4.0 t4.5 + t5.0' and predicts t5.0 residue
    simulator_pqp
    '''
    
parser = argparse.ArgumentParser('MD Residue with energy loss')
args = parser.add_argument('--log_interval',type = int, default = 2000, help = 'log interval to check error')
args = parser.add_argument('--total_time',type = float, default = 50, help ='total time with respect to 0.001')
args = parser.add_argument('--deltat', type = float, default = 0.001, help = 'time step')
args = parser.add_argument('--epochs', type = int, default = 50, help = 'Number of epochs')
args = parser.add_argument('--seed', type = int, default =  937162211, help = 'reproducibiltiy')
args = parser.add_argument('--alpha', type = float, default = 0.5, help = 'penalty constraint weight, 0 means full residue loss')
args = parser.add_argument('--beta', type = float, default = 0.01, help = 'boltzmann loss weight')
args = parser.add_argument('--gamma', type = float , default = 10.0, help = 'boltzmann coefficient')
args = parser.parse_args()

# =============================================================================
# Hyperparameter setting
# =============================================================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 128
deltat = args.deltat
total_time = int(args.total_time)
alpha = args.alpha
gamma = args.gamma
beta = args.beta
INITIAL_DIM = 2
FINAL_DIM = 1024

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def energy(state):
    return torch.mean((state[:,0] ** 2 -1) ** 2 + state[:,0] + ( 0.5 * state[:,1] ** 2),dim = 1)

class MDModel(nn.Module):
    def __init__(self):
        super(MDModel,self).__init__()
        
        self.conv1d = nn.Sequential(
            nn.Conv1d(INITIAL_DIM, FINAL_DIM, kernel_size = 1,stride = 1),
            )
        
        self.fc1 = nn.Sequential(
                nn.Linear(INITIAL_DIM * 11, 512),
                # nn.Linear(FINAL_DIM * 11,512),
                nn.Softplus(),
                # nn.Tanh(),
                # nn.BatchNorm1d(160), 
                nn.Linear(512,256),
                nn.Softplus(),
                # nn.Tanh(),
                nn.Linear(256,64),
                # nn.Softplus(),
                nn.Tanh(),
                nn.Linear(64,2),
            )
       
  
        self.energyLayer = energyLayer()
        
    def forward(self, x):
        #initialize the pq difference layer for each batch
        self.pqDifference = pqDifference(x[:,:,10])

        #the reference energy is calculated using past trajectory without any reference to the
        #predicted trajectory, this helps to smoothen energy fluctuation that may be inherent
        #in the predicted state, however there will be trade off in accuracy of p,q 
        #since the original p, q is calculated using MD that has energy fluctuation
       
        true_energy = energy(x[:,:,:10])
        
        # x = self.conv1d(x.squeeze())
        x = x.reshape(-1,np.prod(list(x.shape[1:])))
        predicted_state = self.fc1(x)

        # we can assume the predicted state to be n x 1 x 2 for n states
        
        predicted_energy = energy(predicted_state.unsqueeze(2)) 
        # print('predicted energy: ', true_energy[0], 'true energy :', predicted_energy[0])
   
        #the output of the model would be dq, dp, dE
        dE = true_energy - predicted_energy
        return (self.pqDifference(predicted_state), dE)
        
def residue_loss(predicted_residue,true_residue):
    criterion = nn.MSELoss(reduction = 'sum')
    # criterion = nn.SmoothL1Loss(reduction = 'sum')
    return criterion(predicted_residue,true_residue)
   
def loss_function_energy(energy_diff):
    # the lost function is adjusted such that the energy difference should be zero for all states

    loss = -gamma * (energy_diff.abs() * -beta).exp() # boltzmann loss
    loss = loss.sum()
    # criterion = nn.SmoothL1Loss(reduction = 'sum')
    # loss = criterion(energy_diff, torch.tensor([0.]).to(device))
    return loss

def combined_loss(true_residue,predicted_residue,energy_diff):
    return alpha * loss_function_energy(energy_diff) + (1-alpha) * residue_loss(predicted_residue, true_residue)
 
    
model = MDModel().to(device)
# optimizer = optim.AdamW(model.parameters(), lr = 1e-4, weight_decay = 0.1) #adam with weight decay
optimizer = optim.Adam(model.parameters(), lr = 1e-3, amsgrad = True)
# scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-3, max_lr=1e-2,cycle_momentum = False)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',patience = 10)

dataset_train = create_dataset(mode = 'train', total_time = total_time)
kwargs = {'pin_memory' : True, 'num_workers' : 6, 'shuffle' : True}
train_loader = DataLoader(dataset_train, batch_size = batch_size, **kwargs)

dataset_validation = create_dataset(mode = 'validation', total_time = total_time)
valid_loader = DataLoader(dataset_validation, batch_size = batch_size , **kwargs)

#test with random data
with torch.no_grad() : 
    data, residue, true_state = next(iter(train_loader))
    predicted_residue, energy_diff = model(data.to(device))
    
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, true_residue, true_state ) in enumerate(train_loader): 
        # =====================================================================
        # here the data is in the form of initial state + large final step state
        # true_state is the true final state with smaller small step 
        # Data is transposed due to convolution reason so that each state will be convolved together
        # =====================================================================
        data.requires_grad_(True)
        data = data.to(device)
        true_residue = true_residue.to(device)
        optimizer.zero_grad()
        predicted_residue, energy_diff = model(data)
         
        
        loss = combined_loss(true_residue, predicted_residue,energy_diff)
        loss.backward()
        
        train_loss += loss
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch : {} [{}/{} ({:.2f} %)] \t Loss : {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss / len(data),
                ))
        scheduler.step(loss)
    print('====> Epoch : {} Average Loss : {:.10f}'.format(
        epoch, train_loss / len(train_loader.dataset)
        ))
    train_loss /= len(train_loader.dataset)
  
    return train_loss
    
def valid(epoch):
    model.eval()
    test_loss = 0
    res_loss = 0
    dE = 0
    with torch.no_grad():
        for batch_idx, (data,true_residue, _ ) in enumerate(valid_loader):
            data = data.to(device)
            true_residue = true_residue.to(device)
            predicted_residue, energy_diff = model(data)
            
            data = data.squeeze()       
            loss = combined_loss(true_residue, predicted_residue, energy_diff)
            test_loss += loss
            
            loss_residue = residue_loss(predicted_residue, true_residue)
            dE += torch.sum(torch.abs(energy_diff))
            res_loss += loss_residue
            
            if batch_idx % args.log_interval == 0:
                print('Test Epoch : {} [{}/{} ({:.2f} %)] \t Loss : {:.6f}'.format(
                    epoch, batch_idx * len(data), len(valid_loader.dataset),
                    100. * batch_idx / len(valid_loader),
                    loss / len(data),
                    ))
            
    test_loss /= len(valid_loader.dataset)
    res_loss /= len(valid_loader.dataset)
    dE /= len(valid_loader.dataset)
    print('====> Validation Test Loss : {:.6f}'.format(test_loss))
    print('====> Validation Position / Velocity Test Loss : {:.10f}'.format(res_loss))
    print('====> Difference in energy : {:.10f} \n'.format(dE))

    return test_loss
    
def test(test_loader):
    
    criterion = nn.SmoothL1Loss(reduction = 'sum')
    epoch_loss = 0
    
    model.eval()
    l1error_position = 0
    l1error_position_2 = 0 # expectation of delta q ^ 2
    
    l1error_velocity = 0
    l1error_velocity_2 = 0 # expectation of delta p ^ 2
    
    dE = 0
    dE_2 = 0
    with torch.no_grad():
        for batch_idx, (data, true_residue, true_state) in enumerate(test_loader):
            data = data.to(device)
            true_residue = true_residue.to(device)
            true_state = true_state.to(device)
            predicted_residue, energy_diff = model(data)
            
            data = data.squeeze().detach().cpu().numpy()
            predicted_residue = predicted_residue.detach().cpu().numpy()
            true_residue = true_residue.detach().cpu().numpy()
            true_state = true_state.detach().cpu().numpy()
            
            dE += torch.sum(torch.abs(energy_diff))
            dE_2 += torch.sum(torch.abs(energy_diff) ** 2)
            
            predicted_state = data[:,:,-1] + predicted_residue

            position_diff = np.abs(predicted_state[:,0] - true_state[:,0])
            velocity_diff = np.abs(predicted_state[:,1] - true_state[:,1])
            
            l1error_position += np.sum(position_diff)
            l1error_velocity += np.sum(velocity_diff)
            
            l1error_position_2 += np.sum(position_diff ** 2.0)
            l1error_velocity_2 += np.sum(velocity_diff ** 2.0)
            
            loss = criterion(torch.tensor(predicted_state),torch.tensor(true_state))
            epoch_loss += loss.item()
            
            print('predicted state : {}, true state : {}'.format(predicted_state[0],true_state[0]))
    
    pos_diff_expectation = l1error_position / len(test_loader.dataset)
    vel_diff_expectation = l1error_velocity / len(test_loader.dataset)
    pos2_diff_expectation = l1error_position_2 / len(test_loader.dataset)
    vel2_diff_expectation = l1error_velocity_2 / len(test_loader.dataset)
    dE /= len(test_loader.dataset)
    dE_2 /= len(test_loader.dataset)
    
    print('L1 Average error position : {:.6f}'.format(pos_diff_expectation)) 
    print('position error variance : {:.6f}'.format(pos2_diff_expectation - pos_diff_expectation ** 2.0)) 
    
    print('L1 Average error velocity : {:.6f}'.format(vel_diff_expectation))
    print('velocity error variance : {:.6f}'.format(vel2_diff_expectation - vel_diff_expectation ** 2.0)) 
    
    print('Average Energy diff : {:.6f}'.format(dE))
    print('Energy error variance : {:.6f}'.format(dE_2 - dE ** 2.0)) 
    
    epoch_loss /= len(test_loader.dataset)
    
    return epoch_loss 


if __name__ == '__main__':
    print('MD Training')
    train_error_list = []
    test_error_list = []
    for epoch in range(1, args.epochs + 1):
        train_error_list.append(train(epoch))
        test_error_list.append(valid(epoch))
    print('Smooth L1 Loss pos/vel : {:.4f} '.format(test(valid_loader)))
    
    currdir = os.getcwd()
    
    torch.save(model.state_dict(), currdir + '/compiled/MD_ML_conv_lasso_energy_loss_01.pth' )
    fig = plt.figure(figsize = (10,10))
    plt.plot(train_error_list, label = 'train loss')
    plt.plot(test_error_list, label = 'validation loss')
    plt.legend(loc = 'best')
