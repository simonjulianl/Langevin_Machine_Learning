#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 13:33:35 2020

@author: simon
"""

import numpy as np
import initialize
import temperature
import force
from tqdm import trange
import matplotlib.pyplot as plt

class Simulation_PQP:
    def __init__(self,N = 2,DIM = 2,BoxSize = 1.0,deltat = 0.2,Nsteps = 5000, Trequested = 1.0, DumpFreq = 1):
        self.DIM = DIM
        self.N = N
        self.BoxSize = BoxSize
        self.volume = self.BoxSize ** DIM
        self.density = N/self.volume
        self.deltat = deltat
        self.Nsteps = Nsteps
        self.Trequested = Trequested
        self.DumpFreq = DumpFreq
        self.temperature = np.ones(self.Nsteps) #instantenous values
        self.ene_pot_aver = np.ones(self.Nsteps)
        self.ene_kin_aver = np.ones(self.Nsteps)
        self.counter = 0
        
        self.kB = 1
        self.u = 1
        
    def set_scaling(self,stat):
        self.scale = stat
        
    def set_mass(self,mass): #in this MSMC, we assume all particles are identical
        self.mass = mass 
      
    def initialize_pos(self, scale = 1 ,file = False):            
        self.pos = initialize.init(self.N,self.DIM,self.BoxSize,file,scale = scale,shift = False)
        self.pos -= 0.5
        self.pos *= 2 * scale #the range is [-scale,scale]
        
    def initialize_vel(self,scale = 1,file = False,v_scaling = True):
        self.vel = initialize.init(self.N,self.DIM,self.BoxSize,file,scale = scale,vel = False,shift = False) # velocity scaling to adjust the temperature
        
        if self.N == 1: # since if there is only one particle v = 0 will lead to error
            self.vel = self.vel + 0.5 # just some arbitrary number since it would be rescaled
            
        if v_scaling : 
            iTemp,_ = self.calculate_temp() #instantenous temperature
            self.lmbda = np.sqrt(self.Trequested/iTemp)
            self.vel = np.multiply(self.vel,self.lmbda) 

    def calculate_temp(self):
        return temperature.temp(self.N,self.DIM,self.BoxSize,self.vel,self.mass)
    
    def log_error(self):
        filename = 'log_file.txt'
        with open(filename,'w') as f:
            upper_range = min(100,len(self.velbefore))
            for i in range(-upper_range,-1,1):
                f.write('{}'.format(self.velbefore[i][0]))
                f.write('|vel before : {:.4E}'.format(self.velbefore[i][1][0].item()))
                f.write('\t\t| vel after : {:.4E}'.format(self.velafter[i][1][0].item()))
                f.write('\t\t| pos before : {:.4E}'.format(self.posbefore[i][1][0].item()))
                f.write('\t\t| pos after : {:.4E}'.format(self.posafter[i][1][0].item()))
                f.write('\n')
        
    def simulate(self, track = False):
        #container to record all the position and velocity of the particles
        self.xlist_all = [[] for i in range(self.N)]
        self.ylist_all = [[] for i in range(self.N)]
        self.zlist_all = [[] for i in range(self.N)]
        
        self.vxlist_all = [[] for i in range(self.N)]
        self.vylist_all = [[] for i in range(self.N)]
        self.vzlist_all = [[] for i in range(self.N)]
        
        if track : 
            self.velbefore = []
            self.velafter = []
            self.posbefore = []
            self.posafter = []
        
        #initialize acceleration based on potential energy
        acc, _ = force.force_double_well(self.N, self.DIM, self.BoxSize, self.pos)
        
        if(self.DIM == 1):
            for i in range(self.N):
                self.xlist_all[i].append(self.pos[:,0][i])
                self.vxlist_all[i].append(self.vel[:,0][i])
                
        #attach the first position and the next N integration 
                
        for k in trange(self.Nsteps,desc = 'internal simulation'):
            #PQP Scheme
            self.temperature[k], self.ene_kin_aver[k] = self.calculate_temp()
            
            #kB= 1, mass = 1
            #udpate P (momentum part) 
            if track : 
                self.velbefore.append((1,self.vel))
                self.posbefore.append((1,self.pos))
            self.vel = self.vel +self.deltat / 2 * acc
            if track :
                self.velafter.append((1,self.vel))
                self.posafter.append((1,self.pos))
                if np.isnan(self.vel[0]) : 
                    self.log_error()
                    break
            #update Q (position part)
            if track:
                self.velbefore.append((2,self.vel))
                self.posbefore.append((2,self.pos))
            self.pos = self.pos + self.deltat * self.vel 
            if(k%self.DumpFreq==0): 
                if(self.DIM == 1):
                    for i in range(self.N):
                        self.xlist_all[i].append(self.pos[:,0][i])
                        self.vxlist_all[i].append(self.vel[:,0][i])
                #if its in 2 dimension, we will use matplotlib to plot the particles
                # if(self.DIM==2):
                #     for i in range(self.N):    
                #         self.xlist_all[i].append(self.pos[:,0][i])
                #         self.ylist_all[i].append(self.pos[:,1][i])
                        
                # #if the particles are in 3D, we will use mplot3d library to plot the position
                # if (self.DIM == 3):
                #     for i in range(self.N):
                #         self.xlist_all[i].append(self.pos[:,0][i])
                #         self.ylist_all[i].append(self.pos[:,1][i])
                #         self.zlist_all[i].append(self.pos[:,2][i])
            if track : 
                self.velafter.append((2,self.vel))
                self.posafter.append((2,self.pos))
                if np.isnan(self.vel[0]) : 
                    self.log_error()
                    break
            #update P (momentum part)
            if track:
                self.velbefore.append((3,self.vel))
                self.posbefore.append((3,self.pos))
            acc, self.ene_pot_aver[k]= force.force_double_well(self.N, self.DIM, self.BoxSize, self.pos,self.scale)
            self.vel = self.vel + self.deltat / 2 * acc
            if track : 
                self.velafter.append((3,self.vel))
                self.posafter.append((3,self.pos))
                if np.isnan(self.vel[0]) :
                    self.log_error()
                    break
            self.temperature[k], self.ene_kin_aver[k] = self.calculate_temp()
            
    def plot_stat(self):   
        total_energy = self.ene_kin_aver + self.ene_pot_aver
        print('Energy fluctuation : ', max(total_energy) - min(total_energy))
        figure = plt.figure(figsize = [20,30])
        figure.suptitle('Data Figure | Timestep : {} '.format(self.deltat),fontsize = 30)
        plt.rc('xtick',labelsize = 20)
        plt.rc('ytick',labelsize = 20)
        
        ax = figure.add_subplot(311)
        ax.plot(total_energy[::],'k-')
        ax.set_ylabel('Total Energy',fontsize = 20)
        ax.set_xlabel("iteration",fontsize = 20)
        ax.set_ylim(-0.93,-0.875)
        # plt.show()
        # plt.subplot(5,1,2)
        # plt.plot(self.temperature,'k-')
        # plt.ylabel("Instantenous Temp",fontsize = 20)
        # plt.xlabel("iteration",fontsize = 20)
        # figure2 = plt.figure(figsize = [5,5])
        ax2 = figure.add_subplot(312)
        ax2.plot(self.ene_pot_aver)
        ax2.set_ylabel('E_P',fontsize = 20)
        ax2.set_xlabel('iteration', fontsize = 20)
        # ax2.set_ylim(min(self.ene_pot_aver), max(self.ene_pot_aver))
        
        # ax2 = figure2.add_subplot(212)
        # plt.scatter(np.array(self.xlist_all[0]) * self.BoxSize ,self.ene_pot_aver)
        # plt.ylabel("E_P",fontsize = 20)
        # plt.xlabel("x position",fontsize = 20)
        
        # plt.show()
        # figure3 = plt.figure(figsize = [5,5])
        ax3 = figure.add_subplot(313)
        ax3.plot(self.ene_kin_aver,'k-')
        ax3.set_ylabel("E_K",fontsize = 20)
        ax3.set_xlabel("iteration",fontsize = 20)
        # ax3.set_ylim(min(self.ene_kin_aver)/10,max(self.ene_kin_aver) * 1.5)
        # figure.savefig('Data_figure.png',dpi = 300 , bbox_inches = 'tight')
        plt.show()

