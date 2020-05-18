#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 18:18:54 2020

@author: simon
"""

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset,TensorDataset,DataLoader
from torchvision import datasets,transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm,trange

import os
from pathlib import Path

import sys
curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
initialization = os.path.join(parent_path, 'initial')

sys.path.append(parent_path)
from error.error import *;
from tqdm import trange;
from models import *;


np.random.seed(2)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class simulation_langevin:
    __instance = None;
    kB = 1
    def __init__(self, time_step : float, Nsteps : int, gamma : float, Temperature : float, Func = None, **kwargs):
        try : 
            print('Initializing')
            self.time_step = time_step
            self.Nsteps = Nsteps
            self.gamma = gamma 
            self.Temp = Temperature
            self.Func = Func
            self.load()
            self.randNormal()
            
            if simulation_langevin.__instance != None:
                raise SingletonError('This class is a singleton')
            else :
                simulation_langevin.__instance = self
            
        except InitializationError:
            raise InitializationError('could not initialize')
            
        
        print('Initialization Finished')
        
    def load(self) -> None:
        ''' helper function to load the data, since we are going to get the p and q 
        distribution overtime, 100 independent trajectories should be enough'''
        self.q = np.load(initialization + '/pos_sampled.npy')[:100].squeeze();
        self.p = np.load(initialization + '/velocity_sampled.npy')[:100].squeeze();
        
    def change(self, time_step : float, Nsteps : int, gamma : float, Temperature : float, Func = None) :
        try : 
            print('Changing setting')
            self.time_step = time_step
            self.Nsteps = Nsteps
            self.gamma = gamma 
            self.Temp = Temperature
            self.Func = Func
            self.load()            
        except InitializationError:
            raise InitializationError('could not initialize')
            
        print('Setting changed')
    
    def randNormal(self): 
        '''helper function to set the random for langevin
        using BP method, there are 2 random vectors'''
        random_1 = np.random.normal(loc = 0.0, scale = 1.0, size = 100 * 10000)
        random_2 = np.random.normal(loc = 0.0, scale = 1.0, size = 100 * 10000)
        #max step size is 10000 in this case
        self.random_1 = random_1.reshape(-1,100)
        self.random_2 = random_2.reshape(-1,100)
            
    def integrate(self) -> tuple: # of (q list ,  p list)
        self.load(); #set the p and q
        ''' mass counted as 1 hence omitted '''
        idx = 0 # counter for random, reset every 1000 steps if used
                
        q_list = np.zeros((self.Nsteps + 1,100))
        p_list = np.zeros((self.Nsteps + 1,100))
        
        q_list[0] = self.q
        p_list[0] = self.p
        
        for i in trange(1, self.Nsteps+1):
            self.p = np.exp(-self.gamma * self.time_step / 2) * self.p + np.sqrt(self.kB * self.Temp * ( 1 - np.exp( - self.gamma * self.time_step))) * self.random_1[idx]
            
            for j in range(int(self.time_step/ 0.01)): # repeat with respect to ground truth
                acc = self.get_force()
    
                self.p = self.p + 0.01 / 2 * acc #dp/dt
                
                self.q = self.q + 0.01 * self.p
                
                acc = self.get_force()
                self.p = self.p + 0.01 / 2 * acc
            
            self.p = np.exp(-self.gamma * self.time_step / 2) * self.p + np.sqrt(self.kB * self.Temp * ( 1 - np.exp( - self.gamma * self.time_step))) * self.random_2[idx]
            
            q_list[i] = self.q
            p_list[i] = self.p
            idx += 1
            
        
        return (q_list.T, p_list.T) # transpose to get particle x trajectory 
    
    def integrate_ML(self) -> tuple : 
        self.load(); # set the p and q
        '''mass counted as 1 hence omitted
        the Lpq in velocity verlect is replaced using Hamiltonian Machine Learning'''
        idx = 0 # counter for random, reset every 1000 steps if used
                
        q_list = np.zeros((self.Nsteps+1,100))
        p_list = np.zeros((self.Nsteps+1,100))
        
        q_list[0] = self.q
        p_list[0] = self.p
        
        for i in trange(1,self.Nsteps+1):
            self.p = np.exp(-self.gamma * self.time_step / 2) * self.p + np.sqrt(self.kB * self.Temp * ( 1 - np.exp( - self.gamma * self.time_step))) * self.random_1[idx]
            
            self.q, self.p = self.velocity_verlet_ML(self.q, self.p) # Lpq is replaced by ML
            
            self.p = np.exp(-self.gamma * self.time_step / 2) * self.p + np.sqrt(self.kB * self.Temp * ( 1 - np.exp( - self.gamma * self.time_step))) * self.random_2[idx]
            
            q_list[i] = self.q
            p_list[i] = self.p
            idx += 1
            
        
        return (q_list.T, p_list.T)
        
    
    def velocity_verlet_ML(self,q, p) -> tuple: # return tuple of next time step
        '''here there is periodic boundary condition since we are using double well
        with potential barrier at x= 2 and x = -2, M and Temperature = 1 by default for simplicity '''
        #here p and q is 1 dimensional 
        q = torch.tensor(q, dtype = torch.float32).to(device).unsqueeze(1).to(device)
        p = torch.tensor(p,dtype = torch.float32).to(device).unsqueeze(1).to(device)
        q.requires_grad = True
        p.requires_grad = True

        hamiltonian = self.Func(p,q) # we need to sum because grad can only be done to scalar
        dpdt = -grad(hamiltonian.sum(), q, create_graph = True)[0] # dpdt = -dH/dq

        #if the create graph is false, the backprop will not update the it and hence we need to retain the graph
        p_half = p +  dpdt * self.time_step / 2 
        
        hamiltonian = self.Func(p_half, q)
        dqdt = grad(hamiltonian.sum(), p, create_graph = True)[0] #dqdt = dH/dp
        q_next = q + dqdt * self.time_step
        
        hamiltonian = self.Func(p_half, q_next)
        dpdt = -grad(hamiltonian.sum(), q, create_graph = True)[0] # dpdt = -dH/dq
    
        p_next = p_half + dpdt * self.time_step  / 2
        
        q_next = q_next.cpu().detach().numpy().squeeze()
        p_next = p_next.cpu().detach().numpy().squeeze()
  
        return (q_next, p_next) # all data is arrange in q p manner
    
    def get_force(self):
        ''' double well potential and force manually computed 
        just simple code for 1 Dimension without generalization'''

        acc = np.zeros([100])
        for i in range(100):
            q = self.q[i]
            phi = (q ** 2.0 - 1) ** 2.0 + q # potential
            dphi = (4 * q) * (q ** 2.0 - 1) + 1.0 # force = dU / dq
            acc[i] = acc[i] - dphi
            

        return acc
    
    def integrate_derivative_ML(self, model_dqdt, model_dpdt) -> tuple : 
        self.load(); # set the p and q
        '''mass counted as 1 hence omitted
        the Lpq in velocity verlect is replaced using Hamiltonian Machine Learning'''
        idx = 0 # counter for random, reset every 1000 steps if used
                
        q_list = np.zeros((self.Nsteps+1,100))
        p_list = np.zeros((self.Nsteps+1,100))
        
        q_list[0] = self.q
        p_list[0] = self.p
        

        for i in trange(1,self.Nsteps+1):
            self.p = np.exp(-self.gamma * self.time_step / 2) * self.p + np.sqrt(self.kB * self.Temp * ( 1 - np.exp( - self.gamma * self.time_step))) * self.random_1[idx]
            
            q = torch.tensor(self.q, dtype = torch.float32).to(device).unsqueeze(1).to(device)
            p = torch.tensor(self.p,dtype = torch.float32).to(device).unsqueeze(1).to(device)
            q.requires_grad = True
            p.requires_grad = True
            
            hamiltonian = model_dpdt(p,q)
            dpdt = -grad(hamiltonian.sum(), q, create_graph = True)[0]
            self.p = self.p + self.time_step / 2 * dpdt.cpu().detach().numpy().squeeze()
            
            q = torch.tensor(self.q, dtype = torch.float32).to(device).unsqueeze(1).to(device)
            p = torch.tensor(self.p,dtype = torch.float32).to(device).unsqueeze(1).to(device)
            q.requires_grad = True
            p.requires_grad = True
            
            hamiltonian = model_dqdt(p,q)
            dqdt = grad(hamiltonian.sum(), p, create_graph = True)[0]
            self.q = self.q + self.time_step * dqdt.cpu().detach().numpy().squeeze()
            
            q = torch.tensor(self.q, dtype = torch.float32).to(device).unsqueeze(1).to(device)
            p = torch.tensor(self.p,dtype = torch.float32).to(device).unsqueeze(1).to(device)
            q.requires_grad = True
            p.requires_grad = True
            
            hamiltonian = model_dpdt(p,q) # be careful , this is special case when i only use 1 input
            dpdt = -grad(hamiltonian.sum(), q, create_graph = True)[0]
            self.p = self.p + self.time_step / 2 * dpdt.cpu().detach().numpy().squeeze()
            
            self.p = np.exp(-self.gamma * self.time_step / 2) * self.p + np.sqrt(self.kB * self.Temp * ( 1 - np.exp( - self.gamma * self.time_step))) * self.random_2[idx]
            
            q_list[i] = self.q
            p_list[i] = self.p
            idx += 1
            
        return (q_list.T, p_list.T)
    
    @staticmethod
    def plot_distribution(trajectory : list, temperature : float) -> None:
        ''' accept a list of trajectory and then plot it
        since there is potential barrier ar x ~ -2 and x ~U = function(torch.tensor([X[i][j]], dtype = torch.float32).to(device), torch.tensor([Y[i][j]], dtype = torch.float32).to(device)) 2 
        I can set the range for dictionary '''
        
        beta = 1 / (simulation_langevin.kB * temperature) # kB is assumed to be unit value 
        
        # p = np.linspace(10,10,100)
        # prob_p = np.exp(-beta * p ** 2.0 / 2)
        # plt.plot(p, prob_p, label = "exact  p integration")
        # plt.show()
        
        q = np.linspace(-2,2,100)      
        prob = np.exp(-beta * ((q ** 2.0 - 1) ** 2.0 + q))
        
        dq = np.array([(q2-q1) for q2,q1 in zip(q[1:],q[:-1])]) # trapezium approximation
        ys = np.array([(y2+y1) / 2. for y2,y1 in zip(prob[1:],prob[:-1])])
        Z = np.dot(ys.T, dq) # for pdf , total area

        hist = None
        bins = None
        for traj in trajectory : 
            n, x = np.histogram(traj, bins = np.linspace(-2,2,100), density = True)
            if hist is None:
                hist = n
            else : 
                hist += n
                
            if bins is None:
                bins = x;
                

        hist /= len(trajectory) # average
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        plt.plot(bin_centers, hist, color = "black", label ="Integration") # use centre instead of edges
        
        plt.plot(q,prob/Z,marker = None, color = "orange", linestyle = '-',label = 'exact')
        plt.legend(loc = "best")
        plt.xlabel("q / position")
        plt.ylabel("pdf")
        plt.show()
            
            
if __name__ == "__main__":
    function = MLP2H_Separable_Hamil(1,10).to(device)
    # function.load_state_dict(torch.load('../MLP2H_Separable_Hamiltonian_05_seed1.pth'))
    temperature = 1
    time_step = 0.5 # the ML Model is at temperature 1
    gamma = 1
    test = simulation_langevin(time_step,10000,gamma,temperature,function);
    
# =============================================================================
#     check gradient
# =============================================================================
    x1 = torch.tensor([-1.545], dtype = torch.float32).to(device)
    x2 = x1 + 0.00001
    x1.requires_grad_(True)
    x2.requires_grad_(True)
    delta_x = x2 - x1
 

    p = torch.tensor([1.0], dtype = torch.float32).to(device)
    p.requires_grad_(True)
    H_x1 = function(p, x1)
    H_x2 = function(p, x2)
    dpdt = (H_x2 - H_x1) / (x2-x1)
        
    real_H = (p ** 2.0 / 2) + ((x1 ** 2.0 - 1 ** 2.0) + x1)
    dpdt_real = grad(real_H, x1, create_graph = True)[0]
    dpdt_derivative = grad(H_x2, x2, create_graph = True)[0]
    print(dpdt, dpdt_derivative)
    print(dpdt_real)
   
    # plt.plot(x, pot, label = "hamiltonian kinetic")
    # plt.plot(x , exact_kinetic , label="exact potential") 
    # plt.legend(loc = "best")
    # plt.ylabel("kinetic")
    # plt.xlabel("p/ momentum")

    # correlation = np.mean(np.abs(result_ML[0] - result[0]),axis =0)
    # print(np.abs(result_ML[0] - result[0]).shape)
 
    # plt.plot(correlation)
    
    # np.save('05_correlation.npy', correlation)
    
    # print(result[0][0])
    # print(result_ML[0][0])
    
    # print(result_ML[0][1] - result[0][1])
# =============================================================================
#   plot KE and U for separate model
# =============================================================================
    separate_model = torch.load('../ML_Hamiltonian05_2models_seed937162211.pth')
    model_dqdt_state = separate_model['model_dqdt']
    model_dpdt_state = separate_model['model_dpdt']
    
    model_dqdt = MLP2H_General_Hamil(2, 10).to(device)
    model_dpdt = MLP2H_General_Hamil(2, 10).to(device)
    
    model_dqdt.load_state_dict(model_dqdt_state)
    model_dpdt.load_state_dict(model_dpdt_state)
    
    result_ML2 = test.integrate_derivative_ML(model_dqdt, model_dpdt)
    q_track  = result_ML2[0]
    simulation_langevin.plot_distribution(q_track, temperature)
    
    result_exact = test.integrate()
    q_track_exact = result_exact[0]
    
    mean_absolute_error = np.mean(np.abs(q_track_exact - q_track) ,axis = 0 )
    plt.plot(mean_absolute_error)
    plt.xlabel("sampling step")
    plt.ylabel("MAE of q")
    plt.title("Mean Absolute Error of q")
    plt.show()
    
    qlist = np.linspace(-4,4,100)
    q = torch.tensor([0.0], dtype = torch.float32).to(device)
    p = torch.tensor([0.0], dtype = torch.float32).to(device)
    q.requires_grad_(True)
    p.requires_grad_(True)
    plist = np.linspace(-10,10,100)
    
    pot = []
    kinetic = []
    for pos in qlist:
        pos = torch.tensor([pos], dtype = torch.float32).to(device)
        potential = model_dpdt(p.unsqueeze(0), pos.unsqueeze(0) )
        
        pot.append(potential.squeeze().cpu().detach().numpy())
        
    for momentum in plist :
        momentum = torch.tensor([momentum], dtype = torch.float32).to(device)
        kin = model_dqdt(momentum.unsqueeze(0) , q.unsqueeze(0))
        kinetic.append(kin.squeeze().cpu().detach().numpy())
        
    plt.plot(qlist, pot ,label = "MD Potential Generator")
    plt.plot(qlist, (qlist ** 2.0 - 1) ** 2.0 + qlist, label = "MD Potential Exact")
    plt.legend(loc = "best")
    plt.ylabel("energy")
    plt.xlabel("q")
    plt.show()
    
    plt.plot(plist, kinetic , label = "MD Kinetic Part")
    plt.plot(plist , (plist ** 2.0 / 2), label = "MD Kinetic Exact")
    plt.legend(loc = "best")
    plt.xlabel("p")
    plt.ylabel("energy")
    plt.show()
    
    
    # particle = 9 # choose from 0- 99 there are 100 particles
    # qtrack_05 = np.load( curr_path + '/MD_SRNN_Langevin/ML05q.npy')[particle]
    # qtrack_001 = np.load( curr_path + '/MD_SRNN_Langevin/MLq.npy')[particle]
    # qtrack_001_exact = np.load( curr_path + '/MD_SRNN_Langevin/exactq.npy')[particle][:len(q_track[particle])]
    
    # # plt.plot(qtrack_05, color = 'black', label = '0.5 track')
    # plt.plot(qtrack_001, color = 'red', label = '0.01 track')
    # plt.plot(q_track[particle], color = "orange" , label = '0.01 2 Models')
    # plt.plot(qtrack_001_exact, color = 'blue', label = '0.01 exact')
    # plt.legend(loc = 'best')
    # plt.show()
    
    # correlation_001_2_model = np.mean(np.abs(result_ML2[0] - result[0]),axis = 0)
    # plt.plot(correlation_001_2_model, color = 'black')
    # plt.show()
    # for i in range(100):
    #     print(result_ML[0][i][-1],result[0][i][-1])
    
    
    # correlation_05 = np.load('05_correlation.npy')
    # correlation_001 = np.load('001_correlation.npy')
    # plt.plot(correlation_05 , color  = 'black' , label = '0.5 correlation /0.01')
    # plt.plot(correlation_001 , color = 'red', label = '0.01 correlation / 0.01')
    # plt.legend(loc = 'best')
    # plt.show()