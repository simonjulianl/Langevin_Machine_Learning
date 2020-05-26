#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 14:59:47 2020

@author: simon
"""

import numpy as np
import initialize
import force
import visualize
from multiprocessing import pool,Process
from tqdm import trange
import time
import error
from scipy.stats import norm
import random
import matplotlib.pyplot as plt
import temperature
from plot_distribution import plot
import copy
import math
import os

'''This is simulation for Molecular Simulation using Monte Carlo to 
generate representatives of configurations using Metropolis Sampling method'''

# =============================================================================
# Currently, this programme is done using arbitrary unit and not real unit
# for practical and experimental purposes
# this is a stand alone code to produce the initial configuration
# =============================================================================

class MSMC:
    global kB
    kB = 1 # arbitrary unit

    def __init__(self,N = 1,DIM = 2,BoxSize = 1.0,iterations = 250000, Temperature = 1.0,DumpFreq = 5,mass = 1):
        self.DIM = DIM
        self.N = N #number of particles, indexed from 0 as listed in the position list 
        self.BoxSize = BoxSize #BoxSize here is just formality for scaling
        self.volume = self.BoxSize ** DIM
        self.density = N / self.volume
        self.Temperature = Temperature
        self.DumpFreq = DumpFreq #this is equivalent to the sampling frequency
        self.iterations = iterations
        self.beta = 1/(kB * self.Temperature)
  
        # =============================================================================
        # Note that proposed distribution needs to be defined
        # =============================================================================
        self.energy_sampled = []
        self.energy_sampled_truncated = []
        self.sampled_configuration_pos = []
        self.sampled_configuration_vel = []
        self.rejection = 0
        self.position = []
        self.velocity = []
        self.counter = 0
        
        self.idx = 0 # track how many have been sampled
        
    def set_mass(self,mass): #in this MSMC, we assume all particles are identical
        self.mass = mass
        
    def initialize_pos(self, scale = 1 ,file = False,shift = False):
        self.pos = initialize.init(self.N,self.DIM,self.BoxSize,file,scale = scale,shift = shift)
        self.pos[0] -= 0.0 # just random guess near the well
        
    def set_delx(self,value): #Attempt to displace a particle
        self.delx = value
    
    def set_delv(self,value): #Attempt to integrate over the momentum dimension
        self.delv = value
        
    def initialize_vel(self,scale = 1,file = False,v_scaling = True,shift = False):
        #by default,it would be random by taking normal distributin - U(0,1)
        #should try to take temperature from maxwell distribution or Gaussian at least
        #shift the centre of mass v, implying external force = 0 for v = 0
        self.vel = initialize.init(self.N,self.DIM,self.BoxSize,file,scale = scale,vel = False,shift = shift) # velocity scaling to adjust the temperature
        
        if self.N == 1:
            self.vel = self.vel + 0.5
            
        if v_scaling : 
            iTemp,_ = self.calculate_temp() #instantenous temperature
            self.lmbda = np.sqrt(self.Temperature/iTemp)
            self.vel = np.multiply(self.vel,self.lmbda)
      
    def set_scaling(self,stat):
        self.scale = stat
     
    def sample_energy(self):
        _, energy = force.force_double_well(self.N,self.DIM,self.BoxSize,self.pos,self.scale)
        self.sampled_configuration_pos.append(self.pos)
        return energy * self.N # total energy 
    
    def sample_momentum(self,scale = 10, mass = 1): #just arbitrary unit6
        #from closed form calculation, the partition function for momentum
        #could be calculated as sqrt(2 * m / beta) * sqrt(pi) * erf(x) from -inf to inf 
        beta = self.beta
        Z = np.sqrt(2 * mass * np.pi / beta ) * 2   
        p = (np.random.rand() - 0.5) * 2.0 * scale # range of drawing samples [-scale,scale]
        pdf = np.exp(-beta * (p ** 2.0) / ( 2 * mass) ) / Z 
        
        plist = np.linspace(-scale,scale,1000)
        sum_prob = np.sum(np.exp(-beta * plist ** 2.0 / (2 * mass)))
        prob = np.exp(-beta * p ** 2.0 / (2 * mass)) / sum_prob
        
        # testing the plot
        # x = np.linspace(-5,5,100)
        # plt.plot(x,np.exp(-beta * (x ** 2.0) / 2)/ sum_prob)
        # plt.xlabel('p')
        # plt.ylabel('prob')
        
        return (prob,p) #return P(p) and p
    
    # this code is to generate position manually, but position could be taken from existing position
    
    # def create_position_predictor(self):
    #     values,bins = plot.distribution(np.array(Sim1.position[burn_in:]), 100)
    #     z = np.polyfit(bins,values,8) #poly fit 8 is chosen based on heuristic value 
    #     self.p = np.poly1d(z)
    #     x_plot = np.linspace(-2,2,100)
    #     plt.plot(x_plot,self.p(x_plot),'k-')
    #     plt.grid(True)
        
    # def sample_position(self,scale = 1) : #this is based on monte carlo integration
    #     x = (np.random.rand() -0.5) * 2.0 * scale # [-scale,scale]
    #     prob = self.p(x)
    #     #tail correction based on the plot
    #     if (x < -1.669) or (x > 1.70) or (prob < 0) : 
    #         prob = 0.0
    #     return (prob,x)
        
    def generate_configuration(self):
        idx = self.idx
        configuration = None
        while configuration is None:
            x = self.position[np.random.randint(0,len(self.position))] # let x already be chosen, choose any random position
            prob_p,p = self.sample_momentum(scale = 10)
            
            alpha = np.random.rand()
            
            if alpha <= (prob_p):
                configuration = (x,p) #tuple of position and momentum
            self.idx += 1;
        
        return configuration
        
    def get_random(self):
        if self.counter == len(self.random):
            self.generate_random()
            self.counter = 0
        random_number = self.random[self.counter]
        self.counter += 1
        return random_number
    
    def generate_random(self):
        self.random = np.random.rand(10000)
        
    def mcmove(self,integrate_vel = False): #particle random displacement. The momentum could be calculated using closed form for 1 particle
        
        o = random.randint(0,len(self.pos) - 1)
        
        #eno is the old potential energy configuration
        _, eno_p = force.force_double_well(self.N, self.DIM, self.BoxSize, self.pos,self.scale)
        
        if integrate_vel:
            _, eno_k = self.calculate_temp()
        else:
            eno_k = 0
            
        eno = eno_p + eno_k
        #let the proposed distribution be a random walk in a L2-euclidian ball of radius r

        xn = np.array(self.pos[o]) + (self.get_random()-0.5) * self.delx
        xo = np.array(self.pos[o])
        self.pos[o] = xn
        
        if integrate_vel :
            vn = np.array(self.vel[o]) + (self.get_random()-0.5) * self.delv
            vo = np.array(self.vel[o])
            self.vel[o] = vn
     
        _, enn_p = force.force_double_well(self.N, self.DIM , self.BoxSize, self.pos, self.scale)
        
        if integrate_vel:
            _, enn_k = self.calculate_temp()
        else:
            enn_k = 0
        
        enn = enn_p + enn_k
        if random.uniform(0,1) >= np.exp(-self.beta * (enn-eno)):
            self.rejection += 1
            self.pos[o] = xo
            if integrate_vel : 
                self.vel[o] = vo
            # we reject the move and restore the old configuration
                
    def calculate_temp(self):
        return temperature.temp(self.N,self.DIM,self.BoxSize,self.vel,self.mass)
    
    def simulate(self,integrate_vel = False): #using metropolis sampling
        self.generate_random()
        for i in trange(self.iterations,desc = 'Simulating'):
            self.mcmove(integrate_vel)
            if i % self.DumpFreq == 0 :
                self.position.append(copy.deepcopy(self.pos[0] * self.BoxSize))
                self.velocity.append(copy.deepcopy(self.vel[0] * self.BoxSize))
                energy = self.sample_energy()
                self.energy_sampled.append(energy)
                if energy < 2: #we cut all the energies above 2
                    self.energy_sampled_truncated.append(energy)
        # self.create_position_predictor()
        
if __name__ == '__main__':
    scale_factor = 1
    iterations = int(500000)
    #we ignore the Burning In phase, lets assume the first 250 * 100 = 25000 random walks are irrelevant
    burn_in = 1000
    Sim1 = MSMC(N = 1,DIM = 1, Temperature = 10.0, iterations = iterations)
    
    #set parameters
    Sim1.set_scaling(True)
    Sim1.set_mass(1)
    #A good step (green) size should give the acceptance ratio between 30 and 60%. source : chem.ucsb.edu
    
    Sim1.set_delx(1.5)
    Sim1.set_delv(0.5)
    Sim1.initialize_pos(scale = scale_factor,shift = False)
    Sim1.initialize_vel(scale = scale_factor,v_scaling = True)
    
    #integrate the potential energy probability
    Sim1.simulate(integrate_vel = False)
    print('Acceptance rate : {}'.format(100. - float(Sim1.rejection * 100.) / float(iterations)))
    
    plot.distribution(np.array(Sim1.velocity), 100, label = 'velocity')
    
    Sim1.position = Sim1.position[burn_in:]
    Sim1.velocity = Sim1.velocity[burn_in:]
    assert len(Sim1.position) == len(Sim1.velocity)
    
    index = random.sample([i for i in range(len(Sim1.position))],25000)
    position_sampled = np.array(Sim1.position)[index]
    velocity_sampled = np.array(Sim1.velocity)[index]
    
    fig = plt.figure(figsize = (15,15))
    
    plt.rc('xtick',labelsize = 10)
    plt.rc('ytick',labelsize = 10)
    plt.subplot(311)
    
    plt.title('Random Walk Results on Energy',fontsize = 15)
    plt.plot(Sim1.energy_sampled[burn_in:])
    plt.subplot(312)
    plt.title('Energy Histogram',fontsize = 15)
    plt.hist(Sim1.energy_sampled_truncated[burn_in:],bins = 300)
    plt.subplot(313)
    plot.distribution(np.array(Sim1.position[burn_in:]), 100,label = 'MCMC')
    fig.tight_layout(pad = 3.0,h_pad = 1.0)
    plt.savefig('MCMC.pdf',bbox_inches='tight')
    
    # uncomment this to sample using closed form, this will take longer
    position_sampled = []
    velocity_sampled = []
    for i in range(25000): 
        position,momentum = Sim1.generate_configuration()
        position_sampled.append(np.array(position))
        velocity_sampled.append(np.array(momentum)/Sim1.mass)
    
    np.save('pos_sampled_T10.npy',np.array(position_sampled))
    np.save('velocity_sampled_T10.npy',np.array(velocity_sampled))
    
