#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 21:35:54 2020

@author: simon
"""
import numpy as np
# np.random.seed(2)
def init(N, DIM , BoxSize,file = False,vel = False,charge = False,scale = 1,shift = True):
    #by default, it would be random by taking uniform distribution - U(0,1)
    if not file:
        arr = np.random.rand(N,DIM) * scale
        if vel == True:
            arr = np.full((N,DIM),0.22) #0.22 for LJ rms v so it has T of 300K ~ 2.5 RU
            return arr
        if charge == True:
            q = np.empty(N,dtype = np.float_)
            q[::2] = 1.0
            q[1::2] = -1.0 #alternating lattice
            return q
    else:
        arr = np.zeros([N,DIM])
        arr = np.genfromtxt(file,skip_header = 1) #skip the header
        arr = arr[:N,:DIM] / BoxSize # scale the box to 1 x 1
        
    #shift to its centre of mass, so it starts at the centre
    if shift :
        MassCentre = np.sum(arr,axis = 0) / N 
        for i in range(DIM):
            arr[:,i] = arr[:,i] - MassCentre[i]
        
    return arr
    


