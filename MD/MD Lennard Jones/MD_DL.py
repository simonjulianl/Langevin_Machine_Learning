#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 12:45:59 2020

@author: simon
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import MD_DataLoader
from tqdm import trange
n = 64

class Unet(nn.Module):
    def contracting_block(self,in_channels,out_channels, kernel_size = 3,padding = 1):
        block = torch.nn.Sequential(
                torch.nn.Conv2d(kernel_size = kernel_size,in_channels = in_channels,out_channels = out_channels,padding = padding),
                torch.nn.ELU(),
#                torch.nn.BatchNorm2d(out_channels),
                torch.nn.Conv2d(kernel_size = kernel_size,in_channels = out_channels,out_channels = out_channels,padding = padding),
                torch.nn.ELU(),
#                torch.nn.BatchNorm2d(out_channels),
                torch.nn.Conv2d(kernel_size = kernel_size,in_channels = out_channels,out_channels = out_channels,padding = padding),
                torch.nn.ELU(),
#                torch.nn.BatchNorm2d(out_channels),
                )
        return block
    
    def expansive_block(self,in_channels,mid_channels,out_channels,kernel_size = 3,padding = 1):
        block = torch.nn.Sequential(
                torch.nn.Conv2d(kernel_size = kernel_size, in_channels = in_channels, out_channels = mid_channels,padding = padding),
                torch.nn.ELU(),
#                torch.nn.BatchNorm2d(mid_channels),
                torch.nn.Conv2d(kernel_size = kernel_size,in_channels = mid_channels,out_channels = mid_channels,padding = padding),
                torch.nn.ELU(),
#                torch.nn.BatchNorm2d(mid_channels),
                torch.nn.Conv2d(kernel_size = kernel_size,in_channels = mid_channels,out_channels = mid_channels,padding = padding),
                torch.nn.ELU(),
#                torch.nn.BatchNorm2d(mid_channels),
                torch.nn.ConvTranspose2d(in_channels = mid_channels,out_channels = out_channels,kernel_size = 3, stride = 1, padding = 1,output_padding = 0),
                )
        
        return block
        
        
    def final_block(self,in_channels,mid_channels,out_channels,kernel_size = 3):
        block = torch.nn.Sequential(
                torch.nn.Conv2d(kernel_size = kernel_size, in_channels = in_channels , out_channels = mid_channels, padding = 1),
                torch.nn.ELU(),
#                torch.nn.BatchNorm2d(mid_channels),
                torch.nn.Conv2d(kernel_size = kernel_size , in_channels = mid_channels, out_channels = mid_channels,padding = 1),
                torch.nn.ELU(),
#                torch.nn.BatchNorm2d(mid_channels),
                torch.nn.Conv2d(kernel_size = kernel_size,in_channels = mid_channels,out_channels = out_channels,padding = 1),
#                torch.nn.Tanh(),
#                torch.nn.BatchNorm2d(out_channels)
               
                )
        return block
    
    def __init__(self,in_channels,out_channel):
        super(Unet,self).__init__()
        self.conv_encode1 = self.contracting_block(in_channels = in_channels,out_channels = 64)
        self.conv_maxpool1 = torch.nn.MaxPool2d(kernel_size = 2)
        self.conv_encode2 = self.contracting_block(64,128)
        self.conv_maxpool2 = torch.nn.MaxPool2d(kernel_size = 2)
        self.conv_encode3 = self.contracting_block(128,256)
        self.conv_maxpool3 =torch.nn.MaxPool2d(kernel_size = 2)
        
        self.bottleneck = torch.nn.Sequential(
                torch.nn.Conv2d(kernel_size = 3,in_channels = 256,out_channels = 512,padding = 1),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(512),
                torch.nn.Conv2d(kernel_size = 3,in_channels = 512,out_channels = 512,padding = 1),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(512),
                torch.nn.ConvTranspose2d(in_channels = 512,out_channels = 256,kernel_size = 3,padding = 1,output_padding = 0,stride = 1)
                )
        
        self.conv_decode3 = self.expansive_block(512,256,128)
        self.conv_decode2 = self.expansive_block(256,128,64)
        self.final_layer = self.final_block(128,64,out_channel) # outchannel = 1
        
        self.fc = nn.Sequential(
                nn.Linear(in_features = 3*6*1, out_features = 256),
                nn.ELU(inplace = True),
                nn.Linear(in_features = 256,out_features = 256),
                nn.ELU(inplace = True),
                nn.Linear(in_features = 256,out_features = 512),
                nn.ELU(inplace = True),
                nn.Linear(in_features = 512, out_features = 512),
                nn.ELU(inplace = True),
                nn.Linear(in_features = 512,out_features = 1024),
                nn.ELU(inplace = True),
                nn.Linear(in_features = 1024,out_features = 512),
                nn.ELU(inplace = True),
                nn.Linear(in_features = 512,out_features = 256),
                nn.ELU(inplace = True),
                nn.Linear(in_features = 256, out_features = 18),
#                nn.ELU(inplace = True),
                
                )
        
    def crop_and_concat(self,upsampled,bypass,crop = False):
        if crop: 
            c = abs((bypass.size()[2] - upsampled.size()[2]))//2 #diff Y
            d = abs((bypass.size()[3] - upsampled.size()[3]))//2 #diff X
            bypass = F.pad(bypass,(-c,-c,-d,-d)) #crop since bypass is the bigger one
        return torch.cat((upsampled,bypass),1)
    
    
    def forward(self,x):
        #encode
        encode_block1 = self.conv_encode1(x)
#there could be maxpool here but the array 3x6 is too small
        encode_block2 = self.conv_encode2(encode_block1)

        encode_block3 = self.conv_encode3(encode_block2)

        
        bottleneck = self.bottleneck(encode_block3)
        
        decode_block3 = self.crop_and_concat(bottleneck,encode_block3,crop = True)
        cat_layer2 = self.conv_decode3(decode_block3)
        decode_block2 = self.crop_and_concat(cat_layer2,encode_block2,crop = True)
        cat_layer1 = self.conv_decode2(decode_block2)
        decode_block1 = self.crop_and_concat(cat_layer1,encode_block1,crop = True)
        final_layer = self.final_layer(decode_block1)
        return final_layer
#    
#        x = x.view(-1,self.num_flat_features(x))
#        x = self.fc(x)
#        return x
#        
    def num_flat_features(self,x):
        size = x.size()[1:]
        num_features  = 1
        for s in size:
            num_features *= s
        return num_features
        
device = ('cuda' if torch.cuda.is_available() else 'cpu')
MD_Unet = Unet(1,1).to(device)

N = 3
DIM = 3
BoxSize = 10.0
batch_size = 50

dataset = MD_DataLoader.MDdata_loader(N,DIM,BoxSize)
train_loader = DataLoader(dataset,batch_size = batch_size,shuffle = True,num_workers = 3)

lr = 1e-4

optimizer = optim.Adam(MD_Unet.parameters(),lr = lr,betas = (0.9,0.999), amsgrad = True)

def train(epoch):
    
    def adjust_learning_rate(optimizer,epoch,lr):
        lr1 = lr * (0.1 ** (epoch//100))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr1
            
    MD_Unet.train()
    adjust_learning_rate(optimizer,epoch,lr)
#    criterion = nn.L1Loss()
    criterion = nn.MSELoss()
#    criterion = nn.CrossEntropyLoss()
    total_loss = 0
#    optimizer = optim.SGD(MD_Unet.parameters(),lr = 1e-6)
    
    for (curr_pos,true_pos) in train_loader: 
        curr_pos = curr_pos.to(device)
        curr_pos.requires_grad = True
        true_pos = true_pos.to(device) 
        true_pos.requires_grad = True
        optimizer.zero_grad()
        output = MD_Unet(curr_pos)
        output = output.reshape(batch_size,1,DIM,2*N)
        loss = criterion(output,true_pos)
        total_loss += loss
        loss.backward()
        optimizer.step()
    
    loss = total_loss.item() / len(train_loader)
    return loss
        

#
#print(next(iter(train_loader))[1])
#print(MD_Unet(next(iter(train_loader))[0].to(device)))

n_epochs = 100*50

import matplotlib.pyplot as plt
loss_list = []

for i in trange(n_epochs):
    loss = train(i)
    loss_list.append(loss)
    print('average loss: {}'.format(loss))
    
plt.plot(loss_list)
def test():
    MD_Unet.eval()
    test_loss = 0
    
    criterion = nn.L1Loss()
    with torch.no_grad():
        for (curr_pos,true_pos) in test_loader:
            curr_pos = curr_pos.to(device)
            true_pos = true_pos.to(device)
            
            output = model(curr_pos)
            
            test_loss += criterion(true_pos,output).item()
            
        test_loss /= len(test_loader.dataset)
        return test_loss
    

