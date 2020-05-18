#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 11:29:11 2020

@author: simon
"""
import numpy as np
import conversion
import initialize
import temperature
import force
import visualize
from multiprocessing import pool,Process
from tqdm import trange
import time
import error
"""
##################################################################################
this is the code to the Molecular Dynamics simulation using Lennard Jones Potential
all units done in calculation are done in reduced unit which can be converted back to real units
using the given function based on Table given by Basic of Molecular Dynamics Chapter 2

The gas used here is argon, to adjust the value of mass, sigma and epsilon
data could be gathered online for different kind of gass, and this simulation is only
applicable for closed shell systems where strong localized bonds are not formed, hence inert
gas is being used here, other choices include Helium, Xenon, etc
##################################################################################
"""

class Simulation_LJ:
    epsilon = 1
    sigma = 1
    Rcutoff = 2.5
    phicutoff = 4.0/(Rcutoff**12) - 4.0/(Rcutoff **6)
    chi = 1 #berendsen thermostat coefficient
    #average of the balls is going out for N = 8 is ~20 for 10000 steps, meaning 1 / 500 steps
    def __init__(self,N = 10,DIM = 2,BoxSize = 10.0,deltat = 1e-4,Nsteps = 30000,Trequested = 2.5,DumpFreq = 1000):#by default, it is a 2 Dimensional Simulator, with N=2 , BoxSize = 10 sigma
        self.DIM = DIM
        self.N = N
        self.BoxSize = BoxSize
        self.volume = BoxSize ** DIM
        self.density = N / self.volume
        self.deltat = deltat
        self.Nsteps = Nsteps
        self.Trequested = Trequested
        self.DumpFreq = DumpFreq

        self.Q = 1000.0 #this is the relaxation of the dynamics of the friction
        self.gamma = 0 #coefficient of the friction term for Nose-Hoover Thermostat and Langevin
        
        self.event = 0
        self.temperature = np.ones(self.Nsteps) #instantenous values
        self.ene_pot_aver = np.ones(self.Nsteps)
        self.ene_kin_aver = np.ones(self.Nsteps)
    
        real_time_step = conversion.convertSI_time(self.deltat)
        print('initializing...')
        print('cell length: {} sigma'.format(self.BoxSize))
        print('Number of Particles : {}'.format(self.N))
        print("Volume : {} | Density : {} ".format(self.volume,self.density))
        print('Timestep : {} s'.format(real_time_step))
        print('Totalstep : {} | Total Timespan : {} s'.format(self.Nsteps,real_time_step * self.Nsteps))
        print('DumpFreq : {}'.format(self.DumpFreq))
        
    def initialize_pos(self, file = False):
        self.pos = initialize.init(self.N,self.DIM,self.BoxSize,file)
        self.pos_backup = self.pos
        
    def reinitialize_pos(self):
        self.pos = self.pos_backup
        
    def initialize_vel(self,file = False,v_scaling = False):
        #by default,it would be random by taking normal distributin - U(0,1)
        #should try to take temperature from maxwell distribution or Gaussian at least
        #shift the centre of mass v, implying external force = 0 for v = 0
        self.vel = initialize.init(self.N,self.DIM,self.BoxSize,file,vel = True) # velocity scaling to adjust the temperature
        
        if v_scaling : 
            self.temperature[0],self.ene_kin_aver[0] = self.calculate_temp()
            self.lmbda = np.sqrt(self.Trequested/self.temperature[0])
            self.vel = np.multiply(self.vel,self.lmbda)
        self.vel_backup = self.vel
        
    def reinitialize_vel(self):
        self.vel = self.vel_backup
        
    def vel_berendsen(self,temperatures,acc):
        return temperature.berendsen(self.Trequested,self.vel,self.deltat,temperatures,acc,coupling_time = 0.1)
    
    def calculate_temp(self):
        return temperature.temp(self.N,self.DIM,self.BoxSize,self.vel)
    
    def force(self):
        return force.force_lj(self.N,self.DIM,self.BoxSize,self.pos,self.Rcutoff,self.sigma,self.epsilon,self.phicutoff)
    
    def simulate(self,filename = None,thermostat = None):
        path_position = './configuration_data/' + 'log_positon'
        path_velocity = './configuration_data/' + 'log_velocity'
        
        #container to record all the position and velocity of the particles
        self.xlist_all = [[] for i in range(self.N)]
        self.ylist_all = [[] for i in range(self.N)]
        self.zlist_all = [[] for i in range(self.N)]
        
        self.vxlist_all = [[] for i in range(self.N)]
        self.vylist_all = [[] for i in range(self.N)]
        self.vzlist_all = [[] for i in range(self.N)]
        momentum_list = []
 
        #initialize acceleration based on potential energy
        acc,_ = force.force_lj(self.N,self.DIM,self.BoxSize,self.pos,self.Rcutoff,self.sigma,self.epsilon,self.phicutoff)
        
        Q=[1000.0, 1000.0]
        xi=[0.0, 0.0]
        vxi=[0.0, 0.0]
        
        for k in trange(self.Nsteps,desc = 'step'):
            for i in range(self.DIM) : #apply the boundary condition method
                period = np.where(self.pos[:,i] > 0.5)
                self.pos[period,i] = self.pos[period,i] - 1.0
                if len(period[0]) > 0 :
                    self.event += len(period[0])
#                    f.write('particle : {} has left the box from right and enter from left at step : {} \n'.format(period[0],k))
                period = np.where(self.pos[:,i] < -0.5)
                self.pos[period,i] = self.pos[period,i] + 1.0
                if len(period[0]) > 0: 
                    self.event += len(period[0])
#                    f.write('particle : {} has left the box from left and enter from right at step : {} \n'.format(period[0],k))
                #record the event of leaving and entering the simulated cell
                    
            
            if thermostat == 'Berendsen' :
                
                self.temperature[k],self.ene_kin_aver[k] = self.calculate_temp()
                self.vel = self.vel_berendsen(self.temperature[k],acc)
                self.pos = self.pos + self.deltat * self.vel
            
    
                #using berendsen Thermostat formula to adjust the temperature (it acts as a heat bath)
                #update the temperature for chi, tau = 1 to set the velocity to the desire temperature
                #chi = np.sqrt(1 + deltat * (Trequested/temperature[k]-1)) # berendsen Thermosat coefficient where v*  = chi * v after every update
                #the algorithm would break down when temperature = 0 meaning the particles are not moving
                #or the two particles collide which causes the Force Calculation to return NAN as r = 0
                
                acc,self.ene_pot_aver[k] = force.force_lj(self.N,self.DIM,self.BoxSize,self.pos,self.Rcutoff,self.sigma,self.epsilon,self.phicutoff)
                self.vel = self.vel_berendsen(self.temperature[k],acc)
                self.temperature[k],self.ene_kin_aver[k] = self.calculate_temp()
                
            elif thermostat == 'v_scaling' :
                #velocity rescaling formula can be done by adding a new parameter, lmbda = np.sqrt(Trequested / temperature[k]) once
                
                self.temperature[k],self.ene_kin_aver[k] = self.calculate_temp()
                self.lmbda =np.sqrt(self.Trequested/self.temperature[k])
                self.vel = self.vel * self.lmbda + 0.5 * acc * self.deltat
                self.pos = self.pos + self.deltat * self.vel 
                acc,self.ene_pot_aver[k] = force.force_lj(self.N,self.DIM,self.BoxSize,self.pos,self.Rcutoff,self.sigma,self.epsilon,self.phicutoff)
            
                self.temperature[k],self.ene_kin_aver[k] = self.calculate_temp()
                self.lmbda =np.sqrt(self.Trequested/self.temperature[k])
                #every time we have velocity update, we just need up recalibrate the v by scaling it using lmbda
                self.vel = self.vel * self.lmbda + 0.5 * acc * self.deltat
                self.temperature[k],self.ene_kin_aver[k] = self.calculate_temp()
                
            elif thermostat == 'NH': #NV is Nose-Hoover Thermostat 
                self.temperature[k],self.ene_kin_aver[k] = self.calculate_temp()
                #update position
                self.pos = self.pos + self.vel * self.deltat + (acc - self.gamma * self.vel) * (self.deltat ** 2 ) * 0.5
            
                #update velocity to deltat/2
                self.vel = self.vel + self.deltat/ 2 * (acc - self.gamma * self.vel)
                
                #update force
                acc,self.ene_pot_aver[k] = force.force_lj(self.N,self.DIM,self.BoxSize,self.pos,self.Rcutoff,self.sigma,self.epsilon,self.phicutoff)
                
                #update gamma + deltat / 2 twice to match velret algorithm using different ke
                total_ene_kin = self.ene_kin_aver[k] * self.N
                self.gamma = self.gamma + self.deltat * 0.5 * temperature.derivative_NH(total_ene_kin ,self.Q,self.N,self.Trequested)
                
                #update KE using the new velocity
                self.temperature[k],self.ene_kin_aver[k] = self.calculate_temp()
                total_ene_kin = self.ene_kin_aver[k] * self.N
                self.gamma = self.gamma + self.deltat * 0.5 * temperature.derivative_NH(total_ene_kin ,self.Q,self.N,self.Trequested)
                
                self.vel = (self.vel + 0.5 * self.deltat * acc) / (1 + 0.5 * self.deltat * self.gamma)
                
            elif thermostat == 'NHC' : 
                #update temperature and kinetic energy
                self.temperature[k],self.ene_kin_aver[k] = self.calculate_temp()
                total_kin_energy = self.ene_kin_aver[k] * self.N
                
                #update the Nose-Hoover Chain
                self.vel, self.ene_kin_aver[k] = temperature.nhchain(Q,self.Trequested,self.deltat,self.N,vxi,xi,total_kin_energy,self.vel)  
                
                #update the position for deltat/2
                self.pos = self.pos + self.deltat * 0.5 * self.vel
                
                #update the force
                acc,self.ene_pot_aver[k] = force.force_lj(self.N,self.DIM,self.BoxSize,self.pos,self.Rcutoff,self.sigma,self.epsilon,self.phicutoff)
                
                #update the velocity for deltat
                self.vel = self.vel + self.deltat * acc
                
                #update the position for deltat/2
                self.pos = self.pos + self.deltat * 0.5 * self.vel
                
                #update the temperature and kinetic energy and compute the Nose-Hover Chain again
                self.temperature[k],self.ene_kin_aver[k] = self.calculate_temp()
                total_kin_energy = self.ene_kin_aver[k] * self.N
                self.vel, self.ene_kin_aver[k] = temperature.nhchain(Q,self.Trequested,self.deltat,self.N,vxi,xi,total_kin_energy,self.vel)  
                self.temperature[k],self.ene_kin_aver[k] = self.calculate_temp()
                
            elif thermostat == 'Langevin':
                self.temperature[k], self.ene_kin_aver[k] = self.calculate_temp()
                
                #random force denoted by R
                #according to fluctuation dissipation, the standard deviation of the random force is given by
                #np.sqrt(self.deltat * kB * temperature * gamma / mass)
                R = np.random.normal(0,1,self.N * self.DIM).reshape(self.N,self.DIM)

                #kB= 1, mass = 1
                coeff = np.sqrt(self.deltat * self.Trequested * self.gamma )
                self.vel= self.vel + self.deltat / 2 * acc - self.gamma * self.vel + coeff * R
                self.pos = self.pos + self.deltat * self.vel
                #update force
                acc,self.ene_pot_aver[k] = force.force_lj(self.N,self.DIM,self.BoxSize,self.pos,self.Rcutoff,self.sigma,self.epsilon,self.phicutoff)
                self.vel= self.vel + self.deltat / 2 * acc - self.gamma * self.vel + coeff * R
                
                self.temperature[k], self.ene_kin_aver[k] = self.calculate_temp()
            else:
                #assume none of thermostat used, we will use standard velocity verlet 
                self.vel = self.vel + 0.5 * self.deltat * acc
                self.pos = self.pos + self.deltat * self.vel 
                acc,self.ene_pot_aver[k] = force.force_lj(self.N,self.DIM,self.BoxSize,self.pos,self.Rcutoff,self.sigma,self.epsilon,self.phicutoff)
                self.vel = self.vel  + 0.5 * self.deltat * acc
                self.temperature[k],self.ene_kin_aver[k] = self.calculate_temp()
                
                
            momentum_list.append(self.vel.sum(axis = 0))
            
            if(k%self.DumpFreq==0): 
   
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
                        
        momentum_list = np.array(momentum_list)
   
        self.momentumx = momentum_list[:,0]
        self.momentumy = momentum_list[:,1]
        #store all the positions and velocity x,y,z separately
        np.save(path_position +'x.npy',np.array(self.xlist_all)) # this array is in the shape of N * Nsteps matrix
        np.save(path_velocity + 'vx.npy',np.array(self.vxlist_all))
        
        if len(self.ylist_all[0]) > 0:
            np.save(path_position + 'y.npy', np.array(self.ylist_all))
            np.save(path_velocity + 'vy.npy',np.array(self.vylist_all))
        
        if len(self.zlist_all[0]) > 0: 
            np.save(path_position + 'z.npy',np.array(self.zlist_all))
            np.save(path_velocity + 'vz.npy',np.array(self.vzlist_all))
            
        print('Finishing simulation...')
        print('total Steps : {} , timestep : {} (RU)'.format(self.Nsteps,self.deltat))
        print('total time span : {} s'.format(conversion.convertSI_time(self.Nsteps*self.deltat)))
        print('fluctuation: {} epsilon'.format(max(self.ene_pot_aver + self.ene_kin_aver) - min(self.ene_pot_aver + self.ene_kin_aver)))
        
        frame = self.Nsteps//self.DumpFreq
        
        print('total event going out :',self.event)
        print('Visualizing...')
        start_time = time.time()
        p1 = Process(target = visualize.project,args = (self.N,self.DIM,self.BoxSize,frame,self.xlist_all,self.ylist_all,self.zlist_all))
        p1.start()
        p1.join()
        self.plot()
        print('Done, time : {} s'.format(time.time() - start_time))
#        
    def plot(self):
        import matplotlib.pyplot as plt
        realtime = conversion.convertSI_time(self.Nsteps * self.deltat)
        iteration = np.linspace(0,realtime,self.Nsteps) /1e-13
        
      
        total_energy = self.ene_kin_aver + self.ene_pot_aver
        
        figure = plt.figure(figsize = [7,30])
        figure.suptitle('Data Figure | Timestep : {} s'.format(conversion.convertSI_time(self.deltat)),fontsize = 30)
        plt.rc('xtick',labelsize = 20)
        plt.rc('ytick',labelsize = 20)
        
        plt.subplot(5,1,1)
        plt.plot(iteration,self.ene_kin_aver,'k-')
        plt.ylabel("E_K/epsilon", fontsize=20)
        plt.xlabel("time / 1e$^{-13}$ s ",fontsize = 20)

        
        range_kin = max(self.ene_kin_aver)- min(self.ene_kin_aver)
        tick = range_kin / 5
        print(tick)
        print(min(self.ene_kin_aver))
        print(max(self.ene_kin_aver))
        plt.yticks(np.arange(min(self.ene_kin_aver),max(self.ene_kin_aver),tick))

        plt.subplot(5,1,2)
        plt.plot(iteration,self.ene_pot_aver,'k-')
        plt.ylabel("E_P/epsilon",fontsize = 20)
        plt.xlabel("time / 1e$^{-13}$ s ",fontsize = 20)

        plt.yticks(np.arange(min(self.ene_pot_aver),max(self.ene_pot_aver),tick))
        
        plt.subplot(5,1,3)
        plt.plot(iteration,total_energy,'k-')
        plt.ylabel('Total Energy / epsilon',fontsize = 20)
        plt.xlabel("time / 1e$^{-13}$ s ",fontsize = 20)
#        plt.ylim(min(total_energy) - 2 * tick, max(total_energy) + 2 * tick)
#        plt.yticks(np.arange(min(total_energy) - 2 * tick,max(total_energy) + 2 * tick,tick))
        #to have the same scale for y, we use the plt.yticks
        
        plt.subplot(5,1,4)
        plt.plot(iteration,conversion.convertSI_temp(self.temperature),'k-')
        plt.ylabel("Instantenous Temp/ K",fontsize = 20)
        plt.xlabel("time / 1e$^{-13}$ s ",fontsize = 20)
        
        
#        plt.subplot(514)
#        plt.plot(iteration,self.momentumx,'k-')
#        plt.ylabel(r"Momentum X/ (epsilon x mass)$^{(\frac{1}{2})}$ ",fontsize = 20)
#        plt.xlabel('time / 1e$^{-13}$ s', fontsize = 20)
#     
#        plt.subplot(515)
#        plt.plot(iteration,self.momentumy,'k-')
#        plt.ylabel(r"Momentum Y/ (epsilon x mass)$^{(\frac{1}{2})}$ ",fontsize = 20)
#        plt.xlabel('time / 1e$^{-13}$ s', fontsize = 20)
        
        figure.savefig('Data_figure.png',dpi = 300 , bbox_inches = 'tight')
    
    
#        
#plt.xticks(np.arange(0, 51, 5)) 
#plt.yticks(np.arange(0, 11, 1)) 

#generate data
#def generate():
#    for i in range(10):
#        test = Simulation_LJ()
#        filename = 'output_truth' + str(i) + '.txt'
#        test.initialize_pos()
#        test.initialize_vel()
#        test.simulate(filename)
#        #reinitialize the points
#        filename = 'output_data' + str(i) + '.txt'
#        #reconfigurate the setting for the same time span , currently 10x Speedup
#        test.deltat = 1e-3
#        test.Nsteps = 100
#        test.reinitialize_pos()
#        test.reinitialize_vel()
#        test.simulate(filename)
        
if __name__ == '__main__':
#    Sim1 = Simulation_LJ()
#    Sim1.initialize_pos()
#    Sim1.initialize_vel(v_scaling = False)
#    Sim1.simulate(thermostat = 'Langevin')

    np.random.seed(2)
    
    Sim2 = Simulation_LJ(deltat = 1e-4,N = 10, Nsteps = 5000 )
    Sim2.initialize_pos()
    Sim2.initialize_vel(v_scaling = False)
    Sim2.simulate(thermostat = 'Berendsen')
    
    #to calculate L2 error, the data must be of the same length
#    assert len(Sim2.ene_kin_aver) == len(Sim1.ene_kin_aver)
    
#    print('L2_error for kinetic energy : ',error.L2_error(Sim2.ene_kin_aver,Sim1.ene_kin_aver))
#    print('L2_error for potential energy : ',error.L2_error(Sim2.ene_pot_aver,Sim2.ene_pot_aver))
#    TotalEnergy1 = Sim1.ene_kin_aver + Sim1.ene_pot_aver
#    TotalEnergy2 = Sim2.ene_kin_aver + Sim2.ene_pot_aver
#    print('L2_error for total energy : ',error.L2_error(TotalEnergy1,TotalEnergy2))
#    print('L2_error for temperature: ', error.L2_error(Sim1.temperature,Sim2.temperature))