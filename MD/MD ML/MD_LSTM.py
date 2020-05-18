#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 15:11:38 2020

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

parser = argparse.ArgumentParser('MD LSTM')
args = parser.add_argument('--log_interval', type = int, default = 1000, help = 'log interval for loss')
args = parser.add_argument('--total_time', type = float ,default = 5, help = 'Total N steps', metavar = 'N')
args = parser.add_argument('--deltat', type = float ,default = 0.05, help = 'time step')
args = parser.add_argument('--epochs', type = int, default = 10 , help = 'Total Epochs')
args = parser.add_argument('--seed', type = int , default = 1 , help = 'seed for reproducibility')
args = parser.add_argument('--LSTM_layer', type = int, default = 1, help = 'increasing layers of stacked LSTM')
args = parser.parse_args()

torch.manual_seed(args.seed)
batch_size = 1

device = 'cuda' if torch.cuda.is_available() else 'cpu'
total_time = args.total_time
deltat = args.deltat

dataset_train = create_dataset(mode = 'train' , total_time_slow = total_time, slow_time_step = deltat)
kwargs = {'pin_memory' : True, 'num_workers' : 6, 'shuffle' : True}
train_loader = DataLoader(dataset_train, batch_size = batch_size, **kwargs)

dataset_validation = create_dataset(mode = 'validation', total_time_slow = total_time, slow_time_step = deltat)
valid_loader = DataLoader(dataset_validation, batch_size = batch_size , **kwargs)

dataset_test = create_dataset(mode = 'test', total_time_slow = total_time, slow_time_step = deltat)
test_loader = DataLoader(dataset_test, batch_size = batch_size , **kwargs)

INPUT_DIM = 2
HIDDEN_DIM = 6
OUTPUT_DIM = 2

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTM,self).__init__()
        
        self.lstm = nn.LSTM(input_dim, hidden_dim,num_layers = args.LSTM_layer) # this is just stacked LSTM 
        
        self.hidden2output = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(inplace = True),
            nn.Linear(16, output_dim),
            )
        
    def forward(self,x):
        seq = x.shape[1] # 0 is batchsize, 2 is 2 since there are only p and q
        lstm_out, _ = self.lstm(x.view(seq,-1,2))

        output = self.hidden2output(lstm_out.view(seq, -1, HIDDEN_DIM))
        
        return output 
    
model = LSTM(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)
optimizer = optim.Adam(model.parameters() , lr = 1e-3, amsgrad = True)
# optimizer = optim.AdamW(model.parameters() ,lr = 1e-3, weight_decay = 1e-2)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr = 5e-3, max_lr = 5e-2, cycle_momentum = False)
#testing with random data 
with torch.no_grad():
    data,label = next(iter(train_loader))
    output = model(data.to(device))
    
criterion = nn.MSELoss(reduction = 'sum')
#directly minimize the difference between p and p' with q and q' without information energy 

training_loss = []
validation_loss = []

def train(epoch):
    epoch_loss = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        optimizer.zero_grad()
        data,label = data.to(device), label.to(device)
        output = model(data)
        output = torch.transpose(output,0,1)
        
        loss = criterion(output, label)
        loss.backward()
        
        optimizer.step()
        epoch_loss += loss.item()
        
        if batch_idx % args.log_interval == 0:
                print('Train Epoch : {} [{}/{} ({:.2f} %)] \t Loss : {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss / len(data),
                    ))
    # scheduler.step()
    
    epoch_loss /= len(train_loader.dataset)
    
    return epoch_loss

def validation(epoch):
    epoch_loss = 0
    for batch_idx, (data, label) in enumerate(valid_loader):
        data,label = data.to(device), label.to(device)
        output = model(data)
        output = torch.transpose(output, 0, 1)
        
        loss = criterion(output,label)
        
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
    epoch_loss = 0
    for batch_idx, (data,label) in enumerate(test_loader):
        data,label = data.to(device), label.to(device)
        output = model(data)

        output = torch.transpose(output, 0, 1)
        print('original :', data[0])
        print('predicted : ', output[0])
        print('truth : ',label[0])
        
        last_output = output[:,-1,:]
        last_label = label[:,-1,:]
        
        loss = criterion(last_output,last_label)
        epoch_loss += loss.item()
        
    epoch_loss /= len(valid_loader.dataset)
    
    return epoch_loss

if __name__ == '__main__':
    for i in trange(args.epochs, desc = 'epoch'):
        training_loss.append(train(i))
        validation_loss.append(validation(i))
        
    print('L2 Loss pos/vel : {:.4f} '.format(test()))
    
    fig = plt.figure(figsize = (10, 10))
    plt.plot(training_loss, label = 'training_loss')
    plt.plot(validation_loss, label = 'validation_loss')
    plt.legend(loc = 'best')
        

        