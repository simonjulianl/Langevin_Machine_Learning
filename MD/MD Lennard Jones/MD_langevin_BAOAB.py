#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 12:25:03 2020

@author: simon
"""

import numpy as np
import initialize
import temperature
import force
from tqdm import trange
import matplotlib.pyplot as plt
from plot_distribution import plot
'''This is the code for all Langevin Dynamic Integration'''
from multiprocessing import Pool
import os
import argparse
parser = argparse.ArgumentParser('Langevin MD')
args = parser.add_argument('--deltat',type = float, default = 0.25, help = 'time step')
args = parser.add_argument('--fast', type = int, default = int(1), help = 'fast step')
args = parser.add_argument('--Nsteps', type = int, default = int(1e6), help ='total_step')
args = parser.add_argument('--core', type =int,default = int(8), help = 'total cores used for multithreading')
args = parser.parse_args()

# np.random.seed(3)
class Simulation_Langevin:
    def __init__(self,N = 2,DIM = 2,BoxSize = 1.0,deltat = 0.2,Nsteps = 5000, Trequested = 1.0, DumpFreq = 1,fast = int(1e2)):
        self.DIM = DIM
        self.N = N
        self.BoxSize = BoxSize
        self.volume = self.BoxSize ** DIM
        self.density = N/self.volume
        self.deltat = deltat
        self.Nsteps = Nsteps
        self.Trequested = Trequested
        self.DumpFreq = DumpFreq
        self.generate_random()
        self.fast = fast
        # print('initializing...')
        # print('cell length: {} '.format(self.BoxSize))
        # print('Number of Particles : {}'.format(self.N))
        # print("Volume : {} | Density : {} ".format(self.volume,self.density))
        # print('Timestep : {} '.format(self.deltat))
        # print('Totalstep : {} | Total Timespan : {} '.format(self.Nsteps,self.deltat* self.Nsteps))
        # print('DumpFreq : {}'.format(self.DumpFreq))
        
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
        
    def set_gamma(self,gamma): #this is a prerequisite for langevin
        # the NAMD package default gamma is 1/ps
        self.gamma = gamma 
        
    def get_random(self):
        if self.counter == len(self.random):
            self.generate_random()
            self.counter = 0
        random_number = self.random[self.counter]
        self.counter += self.fast
        return random_number
    
    def generate_random(self):
        self.random = np.random.randn(100000,self.N,self.DIM)
        
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
        
    def simulate(self,thermostat = 'Langevin', track = True):
        path_position = './configuration_data/' + 'log_positon'
        path_velocity = './configuration_data/' + 'log_velocity'

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
   
        self.slow = self.Nsteps // self.fast
        for k in trange(self.slow,desc = 'internal simulation'):
            assert thermostat == 'Langevin'
            #OBABO scheme
            self.temperature[k], self.ene_kin_aver[k] = self.calculate_temp()
            
            #random force denoted by R
            #according to fluctuation dissipation, the standard deviation of the random force is given by
            #np.sqrt(self.deltat * kB * temperature * gamma / mass)
            for k in range(self.fast):  
                acc, self.ene_pot_aver[k] = force.force_double_well(self.N, self.DIM, self.BoxSize, self.pos,self.scale)            
                #udpate B (momentum part) 
                self.vel = self.vel +self.deltat / 2 * acc
                    
                #kB= 1, mass = 1
            for j in range(self.fast):  
                #update A (position part)
                self.pos = self.pos + self.deltat / 2 * self.vel 

                R = self.get_random()#get a random normal number ~ N(0,1)
                #constant c1 = exp(-gamma * deltat/2)
                c1 = np.exp(-self.gamma * self.deltat * self.fast/ 2) # c2 = c1 ** 2.0
                #update O (gamma part)
                self.vel = c1 * self.vel + np.sqrt(self.kB * self.Trequested * (1 - c1 ** 2.0)) * self.mass ** (-0.5) * R
                #update A (position part)
                self.pos = self.pos + self.deltat / 2 * self.vel 
                
                if(k%self.DumpFreq==0): 
                    if(self.DIM == 1):
                        for i in range(self.N):
                            self.xlist_all[i].append(self.pos[:,0][i])
                            self.vxlist_all[i].append(self.vel[:,0][i])
                    #if its in 2 dimension, we will use matplotlib to plot the particles
                    if(self.DIM==2):
                        for i in range(self.N):    
                            self.xlist_all[i].append(self.pos[:,0][i])
                            self.ylist_all[i].append(self.pos[:,1][i])
                            
                    #if the particles are in 3D, we will use mplot3d library to plot the position
                    if (self.DIM == 3):
                        for i in range(self.N):
                            self.xlist_all[i].append(self.pos[:,0][i])
                            self.ylist_all[i].append(self.pos[:,1][i])
                            self.zlist_all[i].append(self.pos[:,2][i])
       
            
            for j in range(self.fast):
                #update B (momentum part)
                acc, self.ene_pot_aver[k]= force.force_double_well(self.N, self.DIM, self.BoxSize, self.pos,self.scale)
                self.vel = self.vel + self.deltat / 2 * acc
                
            self.temperature[k], self.ene_kin_aver[k] = self.calculate_temp()
            
        # self.plot_stat()
            
    def plot_stat(self):   
        total_energy = self.ene_kin_aver + self.ene_pot_aver

        figure = plt.figure(figsize = [7,30])
        figure.suptitle('Data Figure | Timestep : {} '.format(self.deltat),fontsize = 30)
        plt.rc('xtick',labelsize = 20)
        plt.rc('ytick',labelsize = 20)
        
        plt.subplot(5,1,1)
        plt.plot(total_energy,'k-')
        plt.ylabel('Total Energy',fontsize = 20)
        plt.xlabel("iteration",fontsize = 20)
   
        plt.subplot(5,1,2)
        plt.plot(self.temperature,'k-')
        plt.ylabel("Instantenous Temp",fontsize = 20)
        plt.xlabel("iteration",fontsize = 20)
        
        plt.subplot(5,1,3)
        plt.scatter(np.array(self.xlist_all[0]) * self.BoxSize ,self.ene_pot_aver[:self.slow])
        plt.ylabel("E_P",fontsize = 20)
        plt.xlabel("x position",fontsize = 20)
        
        # plt.subplot(5,1,4)
        # plt.plot(np.array(self.xlist_all[0]) * self.BoxSize,self.ene_kin_aver,'k-')
        # plt.ylabel("E_K",fontsize = 20)
        # plt.xlabel("x position",fontsize = 20)
        # figure.savefig('Data_figure.png',dpi = 300 , bbox_inches = 'tight')

#multiprocessing function 
def DoWork(simulation):
    simulation.simulate(track = False)
    return (np.array(simulation.xlist_all) * simulation.BoxSize,np.array(simulation.vxlist_all) * simulation.BoxSize)
    
if __name__ == '__main__':
    
    '''To Do 1 dimesional Analysis, the OneD correction should be activated and 
    DIM should be set to 2'''
        
   
    pos = np.load('pos_sampled.npy')
    vel = np.load('velocity_sampled.npy') #this is data for a 1D-particle
    
    final_pos_list = []
    final_vel_list = []
    
    idx = 0
    deltat = args.deltat
    Nsteps = args.Nsteps
    fast_step = args.fast
    pool = Pool(processes = args.core)
    directory = os.path.join(os.getcwd(),'MD_Data')
  
    if not os.path.exists(directory):
        os.mkdir(os.path.expanduser(directory))
        print('directory created at {}'.format(directory))
      
    saving_counter = 0
    
    for i in trange(1,desc = 'simulation batch'):
        Simulation = []
        for j in range(1):
            Sim = Simulation_Langevin(DIM = 1,deltat = deltat,N = 1, Nsteps = Nsteps,fast = fast_step)
        
            Sim.set_scaling(True)
            Sim.set_mass(1)
            Sim.set_gamma(1) 
            #shifting false, hence COM and v is not zero
            Sim.initialize_pos(scale = 2)
            Sim.pos[0] = pos[idx]
            Sim.initialize_vel(scale = 1,v_scaling = True)
            Sim.vel[0] = vel[idx]
            Simulation.append(Sim)
            idx += 1
        results = pool.map(DoWork,Simulation)
        
        
        for x_hist,v_hist in results:
            final_pos_list.append(x_hist)
            final_vel_list.append(v_hist)
            
            np.save(directory + '/x_trajectory{}_slow_BAOAB_1e6.npy'.format(saving_counter),np.array(x_hist))
            np.save(directory + '/v_trajectory{}_slow_BAOAB_1e6.npy'.format(saving_counter),np.array(v_hist))
            saving_counter += 1
        
    pool.close()
    pool.join()
    
    # for i in trange(1):
    #     for j in range(1):
    #         Sim = Simulation_Langevin(DIM = 1,deltat = deltat,N = 1, Nsteps = Nsteps,fast = fast_step)
        
    #         Sim.set_scaling(True)
    #         Sim.set_mass(1)
    #         Sim.set_gamma(1) 
    #         #shifting false, hence COM and v is not zero
    #         Sim.initialize_pos(scale = 2)
    #         Sim.pos[0] = pos[idx]
    #         Sim.initialize_vel(scale = 1,v_scaling = True)
    #         Sim.vel[0] = vel[idx]
    #         x_hist, v_hist = DoWork(Sim)
    #         idx += 1
    
        
    #output the max v and min v
    print('maximum v : ' ,np.max(abs(v_hist[np.where(abs(v_hist) > 0)])))
    print('minimum v : ' ,np.min(abs(v_hist[np.where(abs(v_hist) > 0)])))
    print('maximum pos : ', np.max(abs(x_hist[np.where(abs(x_hist) > 0)])))
    
    # values_v, bins_v = plot.distribution(v_hist, 100,min_interval = min(v_hist)[0] , max_interval = max(v_hist)[0])
    
    
    fig = plt.figure()
    values, bins = plot.distribution(np.hstack(final_pos_list), 100, potfunc = 'double_well_asymmetrical')
    fig.savefig('Exact vs OBABO.png')