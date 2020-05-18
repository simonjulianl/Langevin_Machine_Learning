#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 21:42:09 2020

@author: simon
"""
import numpy as np
import math
import force

kb = 1 # in reduced unit
#NA = 6.022*10**(23)
NA = 1
#define constant if necessary

def temp(N,DIM,BoxSize,vel,mass = 1):
    ene_kin = 0.0
    
    for i in range(N):
        real_vel = BoxSize * vel[i,:]
        ene_kin += 0.5 * np.dot(real_vel,real_vel) # 1/2 v^2

    ene_kin_aver = 1.0 * ene_kin / N
    temperature = 2.0 / DIM * ene_kin_aver # by equipartition theorem
    
    return temperature * mass,ene_kin_aver * mass

def configuration_temp(N,DIM,BoxSize,pos,potential = 'Lennard_Jones',mass = 1,kB = 1,scale = True):
    #configurationa temperature defined as 3NkbT = E[q (derivative of potential)]

    if potential == 'Lennard_Jones':
        acc, _ = force.force_lj(N,DIM,BoxSize, pos, Rcutoff = 2.5, sigma = 1,epsilon = 1,phicutoff = 4.0/(2.5**12) - 4.0/(2.5 **6))
    elif potential == 'double_well':
        acc, _ = force.force_double_well(N, DIM, BoxSize, pos,scale = scale)

    expected_value = np.sum(np.array(pos) * np.array(acc) * (-mass))
    temperature = (2.0 / DIM) * expected_value / N / kB
    return temperature

def berendsen(Trequested,vel,deltat,temperature,acc,coupling_time = 0.1):
    #temperature here is instantenous temperature
    chi = np.sqrt(1 + deltat / coupling_time * ( Trequested / temperature - 1 ) ) 
    vel = vel * chi + 0.5 * deltat * acc
    return vel

def derivative_NH(total_kin,Q,N,temperature,kb = 1):
    derivative_term = 1 / Q * (total_kin - (3*N + 1)/2 * temperature * kb)
    return derivative_term
#Nose Hoover Chain , Follow the algorithm 31 from Frenkel & Smit's book:
# Understanding Molecular simulaton From Algorithms to Applications.tbh, still confused how this hamiltonian works
    
def nhchain(Q,T,dt,nparticles,vxi,xi,ke,vel):
    #the kinetic energy passed here is the total kinetic energy
    dt_2 = dt / 2
    dt_4 = dt_2 / 2
    dt_8 = dt_4 /2
    G2 = (Q[0] * vxi[0] * vxi[0] - T * kb)
    vxi[1] += G2 * dt_4
    vxi[0] *= math.exp(-vxi[1] * dt_8)
    G1 = (2 * ke / NA - 3*nparticles*T*kb) / Q[0]
    vxi[0] += G1 * dt_4
    vxi[0] *= math.exp(-vxi[1] * dt_8)
    xi[0] += vxi[0] * dt_2
    xi[1] += vxi[1] * dt_2
    s = math.exp(-vxi[0] * dt_2)
    vel = np.multiply(s,vel)
    ke*=(s*s)
    vxi[0]*=math.exp(-vxi[1]*dt_8)
    G1=(2*ke/NA-3*nparticles*T*kb)/Q[0]
    vxi[0]+=G1*dt_4
    vxi[0]*=math.exp(-vxi[1]*dt_8)
    G2=(Q[0]*vxi[0]*vxi[0]-T*kb)/Q[1]
    vxi[1]+=G2*dt_4
    return vel,ke / nparticles
    