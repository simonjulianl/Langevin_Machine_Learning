#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 15:20:15 2020

@author: simon
"""


import numpy as np
import initialize

def transform_data(N, DIM , BoxSize ,filepos = False,filevel = False):
    pos = initialize.init(N,DIM,BoxSize,filepos)
    vel = initialize.init(N,DIM,BoxSize,filevel)
    
    tf_data = np.zeros([DIM,2*N])
   
    for i in range(DIM) : 
        tf_data[i][:N] = [pos[j][i] for j in range(N)]
        tf_data[i][N:] = [vel[j][i] for j in range(N)]
    
    #the data is in form [ --r(x) -- -- v(X)-- ]
    #                    [ --r(y) -- -- v(y)-- ]
    #                    [ --r(z) -- -- v(z)-- ]
    return tf_data
        
def read_data(file):
    data = np.genfromtxt(file)
    return data #the data is in 6 x 2N forms, where initial and final configuration is included