#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 21:46:34 2020

@author: simon
"""

import numpy as np
import torch

def force_lj(N,DIM,BoxSize, pos, Rcutoff = 2.5, sigma = 1,epsilon = 1,phicutoff = 4.0/(2.5**12) - 4.0/(2.5 **6)):
    Sij = np.zeros(DIM)
    Rij = np.zeros(DIM)
    
    ene_pot = np.zeros(N) # container for ene_pot
    acc = np.zeros([N,DIM])
    
    for i in range(N-1):
        for j in range(i+1,N):
            Sij = pos[i,:] - pos[j,:]
            #minimum image criterion, we would always choose 1 image to interact with, we choose the closer one
            #since the distance between 1 atom and its image is the length of the cell, we can always revert the sign 
            #of the difference of the distance if another atom is closer to the image instead of the real atom
            # o ------ ( 0.4 * BoxSize) --- o ------(0.6 * BoxSize) ---- o, hence we choose the 0.4 closer
            for k in range(DIM):
                if (np.abs(Sij[k]) > 0.5):
                    Sij[k] = Sij[k] - np.copysign(1.0,Sij[k])

            Rij = BoxSize * Sij 
            Rsqij = np.dot(Rij,Rij) # get square of the real distance
            
            if (Rsqij < Rcutoff ** 2): #check that the distance is within Rcut
                rm2 = sigma /Rsqij # 1/r^2
                rm6 = rm2 ** 3.0  # 1/r^6
                rm12 = rm6 ** 2.0 # 1/r^12
                phi = epsilon * (4.0 * (rm12 -rm6) - phicutoff) # shift up the potential
                dphi = 48.0 / Rsqij * epsilon * (rm12 - 0.5 * rm6)
            
                #dphi is F/r because F(x) = dV/dr * dr/dx = dV/dr * x/r, hence F/r * x = F(x)
                #Since Sij = x/BoxSize, dphi * x / BoxSize will rescale the acceleration to by BoxSize
                #The Lennard Jones ( LJ ) potential accounts for V of 2 particle systems
                #hence each particle will possess half of the potential energy
                
                ene_pot[i] = ene_pot[i] + 0.5 * phi 
                ene_pot[j] = ene_pot[j] + 0.5 * phi
                acc[i,:] = acc[i,:] + dphi * Sij 
                acc[j,:] = acc[j,:] - dphi * Sij

    return acc, np.sum(ene_pot)/N # average of potential energy

def force_double_well(N, DIM, BoxSize, pos,scale = True):
    ene_pot = np.zeros(N)
    acc = np.zeros([N,DIM])

    for i in range(N):
        for k in range(DIM): #calcualte potential per component
            if scale:
                r= pos[i][k] * BoxSize 
            else:
                r = pos[i][k]
            phi = (r ** 2.0 -1) ** 2.0 + r
            dphi = (4 * r) * (r ** 2.0 - 1) + 1.0
            ene_pot[i] += phi
            acc[i,k] = acc[i,k] - dphi
    
    return acc, np.sum(ene_pot)/N
    
def force_symmetrical_well(N, DIM, BoxSize, pos, scale = True):
    ene_pot = np.zeros(N)
    acc = np.zeros([N,DIM])
    
    for i in range(N):
        for k in range(DIM):
            if scale : 
                x = pos[i][k] * BoxSize
            else:
                x = pos[i][k]
        phi = 0.25 * (x-1) ** 2 * (x+1) ** 2 
        dphi = x * (x-1)*(x+1)
        ene_pot[i] += phi
        acc[i,k ] = acc[i,k] - dphi
        
    return acc, np.sum(ene_pot)/N

def force_double_basin2D(N, DIM,  BoxSize, pos, scale = True):
    assert DIM == 2
    ene_pot = np.zeros(N)
    acc = np.zeros([N,DIM])
    
    for i in range(N):
        for k in range(DIM):
            if scale:
                r = pos[i][k] * BoxSize
            else:
                r = pos[i][k]
        x = torch.tensor(float(pos[i,0]),requires_grad = True)
        y = torch.tensor(float(pos[i,1]),requires_grad = True)
        
        phi = (1 + 5 * (torch.exp( -250 * torch.pow((y - 0.25),2)) * torch.exp(-100 * torch.pow(x,2)))) / (torch.pow(torch.pow(x,2) + torch.pow(y,2),0.5) - 1) + 4/5 * torch.exp(-20 * torch.pow(x,2))
        phi.backward()
        ene_pot[i] += phi.item()
        acc[i,0] = acc[i,0] - x.grad.item()
        acc[i,1] = acc[i,1] - y.grad.item()
    
    return acc, np.sum(ene_pot)/N
        
        
    