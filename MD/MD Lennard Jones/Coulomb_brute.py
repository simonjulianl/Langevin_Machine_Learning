#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 16:32:43 2020

@author: simon
"""


#just for fun brute force method of coulomb force

pot_shell = np.zeros(nbox_sq+1,dtype = np.float_)
nbox = 8 
nbox_sq = 64
for x,y,z in product(range(-nbox,nbox+1),repeat = 3):
    #for a box sphere radius of nbox
    rbox = x** 2 + y ** 2 + z ** 2 
    if rbox_sq > nbox_sq :
        continue
    rbox_vec = np.array([x,y,z],dtype = np.float_)
    start_shift = 0 if rbox_sq > 0 else 1
    
    for shift in range(start_shift,n):
        rij = r - np.roll(r,shift,axis = 0) - rbox_vec
        rij_mag = np.sqrt(np.dot(rij,rij))
        pot_shell[rbox_sq] = pot_shell[rbox_sq] + np.sum(q * np.roll(q,shift) / rij_mag)
        
pot_shell = pot_shell / 2.0
pot_shell = np.cumsum(pot_shell)

print('shell potential')
for rbox in range(nbox + 1):
    print(rbox,pot_shell[rbox**2]) # print for sphreical side only