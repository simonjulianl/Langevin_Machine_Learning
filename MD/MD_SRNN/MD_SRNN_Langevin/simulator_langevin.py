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
import multiprocessing

seed = 937162211
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(seed) # apparently this is required
torch.cuda.manual_seed_all(seed) # gpu vars

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class simulation_langevin:
    __instance = None;
    kB = 1
    def __init__(self, time_step : float, Nsteps : int, gamma : float, Temperature : float,  N : float, Func = None, **kwargs):
        try : 
            print('Initializing')
            self.time_step = time_step
            self.Nsteps = Nsteps
            self.gamma = gamma 
            self.Temp = Temperature
            self.Func = Func
            self.N = N
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
        self.q = np.load(initialization + '/pos_sampled.npy')[:self.N].squeeze();
        self.p = np.load(initialization + '/velocity_sampled.npy')[:self.N].squeeze();
        
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
    
    def getCurrDistribution(self) -> None:
        beta = 1 / (simulation_langevin.kB * temperature)
        q = np.linspace(-2,2,1000)      
        prob = np.exp(-beta * ((q ** 2.0 - 1) ** 2.0 + q))
        
        dq = np.array([(q2-q1) for q2,q1 in zip(q[1:],q[:-1])]) # trapezium approximation
        ys = np.array([(y2+y1) / 2. for y2,y1 in zip(prob[1:],prob[:-1])])
        Z = np.dot(ys.T, dq) # for pdf , total area
        
        #plot the distribution of q
        n_q, bins_q = np.histogram(self.q, bins = np.linspace(-2,2,20), density = True)
        #bins chosen from -2 to 2 because at x = 2 at x = -2, prob(x) ~ 0 , they act as potential barrier for T = 1
        binq_centers = 0.5 * (bins_q[1:] + bins_q[:-1])
        plt.plot(q,prob/Z,marker = None, color = "orange", linestyle = '-',label = 'exact') # with reference to exact
        plt.plot(binq_centers, n_q, color = "black" , label = "q0 distribution")
        plt.ylabel("pdf")
        plt.xlabel("q / position ")
        plt.legend(loc="best")
        plt.show()
        
        q_hist = np.array([binq_centers, n_q])
        #plot the distribution of p
        
        p = np.linspace(-4,4,1000)
        prob_p = np.exp(-beta * (p ** 2.0) / 2)
        
        dp = np.array([(p2-p1) for p2,p1 in zip(p[1:],p[:-1])]) # trapezium approximation
        yps = np.array([(y2+y1) / 2. for y2,y1 in zip(prob_p[1:],prob_p[:-1])])
        Zp = np.dot(yps.T, dp)
        
        n_p, bins_p = np.histogram(self.p , bins = np.linspace(-4,4,20), density = True)
        #bins chosen from -4 to 4 because at p = -4 or 4 , prob(p) ~ 0 
        binp_centers = 0.5 * (bins_p[1:] + bins_p[:-1])
        plt.plot(p,prob_p/Zp,marker = None, color = "orange", linestyle = '-',label = 'exact') # with reference to exact
        plt.plot(binp_centers, n_p, color = "blue", label = "p0 distribution")
        plt.ylabel("pdf")
        plt.xlabel("p / momentum")
        plt.legend(loc = "best")
        plt.show()
        
        p_hist = np.array([binp_centers, n_p])
 
    def randNormal(self): 
        '''helper function to set the random for langevin
        using BP method, there are 2 random vectors'''
        random_1 = np.random.normal(loc = 0.0, scale = 1.0, size = self.N * total_step)
        random_2 = np.random.normal(loc = 0.0, scale = 1.0, size = self.N * total_step)
        self.random_1 = random_1.reshape(-1,self.N)
        self.random_2 = random_2.reshape(-1,self.N)
            
    def integrate(self) -> tuple: # of (q list ,  p list)
        self.load(); #set the p and q
        ''' mass counted as 1 hence omitted '''

                
        try :
            q_list = np.load('{}-{}_exactq_{}_seed{}.npy'
                             .format(time_step, ground_truth_step, total_step, seed)) # the first one is the large time step
            p_list = np.load('{}-{}_exactp_{}_seed{}.npy'
                             .format(time_step, ground_truth_step, total_step, seed)) 
            
        except : 
            
            q_list = np.zeros((self.Nsteps + 1,self.N))
            p_list = np.zeros((self.Nsteps + 1,self.N))
            
            q_list[0] = self.q
            p_list[0] = self.p
            
          
            def get_force_helper(q) :
                phi = (q ** 2.0 - 1) ** 2.0 + q # potential
                dphi = (4 * q) * (q ** 2.0 - 1) + 1.0 # force = dU / dq
                acc = -dphi
                return acc
            
            def integrate_helper(q, p, num, return_dict):
                idx = 0 # counter for random, reset every 1000 steps if used
                
                total_particle = 1000 if N >= 1000 else N # total particle per process
                
                q_list_temp = np.zeros((self.Nsteps, total_particle))
                p_list_temp = np.zeros((self.Nsteps, total_particle)) # since there is only 1 particle
                for i in trange(self.Nsteps):
                    p = np.exp(-self.gamma * self.time_step / 2) * p + np.sqrt(self.kB * self.Temp * ( 1 - np.exp( -self.gamma * self.time_step))) * self.random_1[idx][num]
                    
                    for j in range(int(self.time_step/ ground_truth_step)): # repeat with respect to ground truth
                        acc = get_force_helper(q)
                        
                        p = p + ground_truth_step / 2 * acc #dp/dt
                        
                        q = q + ground_truth_step * p
                        
                        acc = get_force_helper(q)
                        p = p + ground_truth_step / 2 * acc
                    
                    p = np.exp(-self.gamma * self.time_step / 2) * p + np.sqrt(self.kB * self.Temp * ( 1 - np.exp( -self.gamma * self.time_step))) * self.random_2[idx][num]
                 
                    q_list_temp[i] = q
                    p_list_temp[i] = p
        
                    idx += 1
                    
                return_dict[num] = (q_list_temp, p_list_temp) # stored in q,p order
               
            assert len(self.p) == len(self.q)
  
            processes = [] # list of processes to be spread
            manager = multiprocessing.Manager()
            return_dict = manager.dict()
           
            for i in range(0,len(self.p),1000):
                p  = multiprocessing.Process(target = integrate_helper, args = (self.q[i:i+1000], self.p[i:i+1000], i, return_dict))
                processes.append(p)
           
            for p in processes :
                p.start()
                
            for p in processes :
                p.join() # block the main thread 
                
            #populate the original q list and p list
            for i in return_dict.keys(): #skip every 1000
                q_list[1:,i:i+1000] = return_dict[i][0] # q
                p_list[1:,i:i+1000] = return_dict[i][1] # p 
      
            np.save('{}-{}_exactq_{}_seed{}.npy'
                             .format(time_step, ground_truth_step, total_step, seed),q_list)
            np.save('{}-{}_exactp_{}_seed{}.npy'
                             .format(time_step, ground_truth_step, total_step, seed),p_list)
            
        finally : 

            return (q_list.T, p_list.T) # transpose to get particle x trajectory 
    
    def integrate_ML(self) -> tuple : 
        self.load(); # set the p and q
        '''mass counted as 1 hence omitted
        the Lpq in velocity verlect is replaced using Hamiltonian Machine Learning'''
        idx = 0 
                
        q_list = np.zeros((self.Nsteps+1,self.N))
        p_list = np.zeros((self.Nsteps+1,self.N))
        
        q_list[0] = self.q
        p_list[0] = self.p
        
        for i in trange(1,self.Nsteps+1):
            self.p = np.exp(-self.gamma * self.time_step / 2) * self.p + np.sqrt(self.kB * self.Temp * ( 1 - np.exp( -self.gamma * self.time_step))) * self.random_1[idx]
            
            self.q, self.p = self.velocity_verlet_ML(self.q, self.p) # Lpq is replaced by ML
            
            self.p = np.exp(-self.gamma * self.time_step / 2) * self.p + np.sqrt(self.kB * self.Temp * ( 1 - np.exp( -self.gamma * self.time_step))) * self.random_2[idx]
            
            q_list[i] = self.q
            p_list[i] = self.p
            idx += 1
            
        
        return (q_list.T, p_list.T)
        
    
    def velocity_verlet_ML(self,q, p) -> tuple: # return tuple of next time step
        #here p and q is 1 dimensional 
        q = torch.tensor(q, dtype = torch.float32).to(device).unsqueeze(1).to(device)
        p = torch.tensor(p,dtype = torch.float32).to(device).unsqueeze(1).to(device)
        q.requires_grad = True
        p.requires_grad = True

        hamiltonian = self.Func(p,q) # we need to sum because grad can only be done to scalar
        dpdt = -grad(hamiltonian.sum(), q, create_graph = False)[0] # dpdt = -dH/dq

        #if the create graph is false, the backprop will not update the it and hence we need to retain the graph
        p_half = p +  dpdt * self.time_step / 2 
        
        hamiltonian = self.Func(p_half, q)
        dqdt = grad(hamiltonian.sum(), p, create_graph = False)[0] #dqdt = dH/dp
        q_next = q + dqdt * self.time_step
        
        hamiltonian = self.Func(p_half, q_next)
        dpdt = -grad(hamiltonian.sum(), q, create_graph = False)[0] # dpdt = -dH/dq
    
        p_next = p_half + dpdt * self.time_step  / 2
        
        q_next = q_next.cpu().detach().numpy().squeeze()
        p_next = p_next.cpu().detach().numpy().squeeze()
  
        return (q_next, p_next) # all data is arrange in q p manner
    
    def get_force(self):
        ''' double well potential and force manually computed 
        just simple code for 1 Dimension without generalization'''

        acc = np.zeros([self.N])
        for i in range(self.N):
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
                
        q_list = np.zeros((self.Nsteps+1,self.N))
        p_list = np.zeros((self.Nsteps+1,self.N))
        
        q_list[0] = self.q
        p_list[0] = self.p
        

        for i in trange(1,self.Nsteps+1):
            self.p = np.exp(-self.gamma * self.time_step / 2) * self.p + np.sqrt(self.kB * self.Temp * ( 1 - np.exp( - self.gamma * self.time_step))) * self.random_1[idx]
            
            q = torch.tensor(self.q, dtype = torch.float32).to(device).unsqueeze(1).to(device)
            p = torch.tensor(self.p,dtype = torch.float32).to(device).unsqueeze(1).to(device)
            q.requires_grad = True
            p.requires_grad = True
            
            hamiltonian = model_dpdt(p,q)
            dpdt = -grad(hamiltonian.sum(), q, create_graph = False)[0]
            self.p = self.p + self.time_step / 2 * dpdt.cpu().detach().numpy().squeeze()
            
            torch.cuda.empty_cache()
            del p,q,hamiltonian, dpdt
            
            q = torch.tensor(self.q, dtype = torch.float32).to(device).unsqueeze(1).to(device)
            p = torch.tensor(self.p,dtype = torch.float32).to(device).unsqueeze(1).to(device)
            q.requires_grad = True
            p.requires_grad = True
            
            hamiltonian = model_dqdt(p,q)
            dqdt = grad(hamiltonian.sum(), p, create_graph = False)[0]
            self.q = self.q + self.time_step * dqdt.cpu().detach().numpy().squeeze()
            
            torch.cuda.empty_cache()
            del p,q, hamiltonian, dqdt
            
            q = torch.tensor(self.q, dtype = torch.float32).to(device).unsqueeze(1).to(device)
            p = torch.tensor(self.p,dtype = torch.float32).to(device).unsqueeze(1).to(device)
            q.requires_grad = True
            p.requires_grad = True
            
            hamiltonian = model_dpdt(p,q) # be careful , this is special case when i only use 1 input
            dpdt = -grad(hamiltonian.sum(), q, create_graph = False)[0]
            self.p = self.p + self.time_step / 2 * dpdt.cpu().detach().numpy().squeeze()
            
            self.p = np.exp(-self.gamma * self.time_step / 2) * self.p + np.sqrt(self.kB * self.Temp * ( 1 - np.exp( - self.gamma * self.time_step))) * self.random_2[idx]
            
            torch.cuda.empty_cache()
            del p,q, hamiltonian, dpdt
            
            q_list[i] = self.q
            p_list[i] = self.p
            idx += 1
                  
        return (q_list.T, p_list.T)
    
    @staticmethod
    def plot_distribution_q(trajectory : list, temperature : float) -> None:
        ''' accept a list of trajectory and then plot it
        since there is potential barrier ar x ~ -2 and x ~U = function(torch.tensor([X[i][j]], dtype = torch.float32).to(device), torch.tensor([Y[i][j]], dtype = torch.float32).to(device)) 2 
        I can set the range for dictionary '''
        
        beta = 1 / (simulation_langevin.kB * temperature) # kB is assumed to be unit value 
            
        q = np.linspace(-4,4,1000)      
        prob = np.exp(-beta * ((q ** 2.0 - 1) ** 2.0 + q))
        
        dq = np.array([(q2-q1) for q2,q1 in zip(q[1:],q[:-1])]) # trapezium approximation
        ys = np.array([(y2+y1) / 2. for y2,y1 in zip(prob[1:],prob[:-1])])
        Z = np.dot(ys.T, dq) # for pdf , total area

        hist = None
        bins = None
        for traj in trajectory : 
            n, x = np.histogram(traj, bins = q, density = True)
            if hist is None:
                hist = n
            else : 
                hist += n
                
            if bins is None:
                bins = x;
                

        hist /= len(trajectory) # average
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        plt.plot(bin_centers, hist, color = "black", label ="Integration ML") # use centre instead of edges
        plt.plot(q,prob/Z,marker = None, color = "orange", linestyle = '-',label = 'exact')
        plt.legend(loc = "best")
        plt.xlabel("q / position")
        plt.ylabel("pdf")
        plt.show()
        np.save('distribution_q_{}using{}_{}_N{}_{}.npy'.
                format(str(time_step).replace('.',''),
                       str(ground_truth_step).replace('.',''),
                       total_step, 
                       N, 
                       hidden_units), np.array((bin_centers, hist)))
        
    @staticmethod
    def plot_distribution_p(trajectory : list, temperature : float ) -> None:
        beta = 1 / (simulation_langevin.kB * temperature)
        
        p = np.linspace(-10,10,100)
        prob_p = np.exp(-beta * p ** 2.0 / 2)
        
        dp = np.array([(p2-p1) for p2,p1 in zip(p[1:],p[:-1])])
        ys = np.array([(y2+y1) / 2. for y2,y1 in zip(prob_p[1:],prob_p[:-1])])
        Z = np.dot(ys.T , dp)
        
        hist = None;
        bins = None;
        
        for traj in trajectory :
            n, x = np.histogram(traj, bins = p, density = True)
            if hist is None:
                hist = n
            else : 
                hist += n
                
            if bins is None:
                bins = x;
        
        hist /= len(trajectory)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        plt.plot(bin_centers, hist, color = "black", label = "Integration ML");
        plt.plot(p, prob_p / Z, color = "orange" , label = "exact")
        plt.legend(loc = "best")
        plt.xlabel("p / position")
        plt.ylabel("pdf")
        plt.show()
        np.save('distribution_p_{}using{}_{}_N{}_{}.npy'.
                format(str(time_step).replace('.',''),
                       str(ground_truth_step).replace('.',''),
                       total_step, 
                       N,
                       hidden_units), np.array((bin_centers, hist)))
    
    @staticmethod 
    def plot_energy(q_list : list, p_list : list) -> None:
        q_list,p_list = np.array(q_list), np.array(p_list)
        potential = (q_list ** 2.0 - 1) ** 2.0  + q_list
        kinetic = (p_list) ** 2.0 / 2
        energy = np.mean((potential + kinetic) , axis = 0)
        plt.plot(energy, label = 'ave energy', color = 'orange');
        plt.xlabel('sampling step')
        plt.ylabel('energy')
        plt.legend(loc = 'best')
        plt.show()
        np.save('energy_{}using{}_{}_N{}_{}.npy'.
                format(str(time_step).replace('.',''),
                       str(ground_truth_step).replace('.',''),
                       total_step, 
                       N,
                       hidden_units), np.array(energy))
        
if __name__ == "__main__":
    function = MLP2H_Separable_Hamil(1,10).to(device)
    # function.load_state_dict(torch.load('../MLP2H_Separable_Hamiltonian_05_seed1.pth'))
    temperature = 1
    time_step = float(input('please input large time step : ')) # the ML Model is at temperature 1
    ground_truth_step = float(input('please input ground truth step : '))
    gamma = 1
    N = int(input('please input num of trajectories : ')) # total number of initial trajectories
    #there is an error with particle 2640, apparently its out of st.  < -2 
    #max combination is around 10000 x 10000
    total_step = int(input('please input number of total step : '))
    test = simulation_langevin(time_step,total_step,gamma,temperature, N, function);
    test.getCurrDistribution()
    
 
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
        
    real_H = (p ** 2.0 / 2) + ((x1 ** 2.0 - 1) ** 2.0 + x1)
    dpdt_real = grad(real_H, x1, create_graph = False)[0]
    dpdt_derivative = grad(H_x2, x2, create_graph = False)[0]
    print(dpdt, dpdt_derivative)
    print(dpdt_real)
   
# =============================================================================
#   plot KE and U for separate model
# =============================================================================
    hidden_units = int(input('please input number of units/hidden layer : '))
    separate_model = torch.load('../ML_Hamiltonian{}_{}_2models_seed937162211_{}_2H.pth'
                                .format(str(time_step).replace('.',''),
                                        str(ground_truth_step).replace('.',''),
                                        hidden_units))
    
    model_dqdt_state = separate_model['model_dqdt']
    model_dpdt_state = separate_model['model_dpdt']
    
    model_dqdt = MLP2H_General_Hamil(2, hidden_units).to(device)
    model_dpdt = MLP2H_General_Hamil(2, hidden_units).to(device)
    
    model_dqdt.load_state_dict(model_dqdt_state)
    model_dpdt.load_state_dict(model_dpdt_state)
    
    result_ML2 = test.integrate_derivative_ML(model_dqdt, model_dpdt)
    q_track, p_track  = result_ML2
    simulation_langevin.plot_distribution_q(q_track, temperature)
    simulation_langevin.plot_distribution_p(p_track, temperature)
    print(q_track, p_track)
    simulation_langevin.plot_energy(q_track, p_track)
    
    result_exact = test.integrate()
    q_track_exact = result_exact[0]
    
    mean_absolute_error = np.mean(np.abs(q_track_exact - q_track) ,axis = 0 )
    np.save('MAE_q_{}using{}_{}_N{}_{}.npy'.
                format(str(time_step).replace('.',''),
                       str(ground_truth_step).replace('.',''),
                       total_step, 
                       N,
                       hidden_units), np.array(mean_absolute_error))
    
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
    
    dUdq = [] # dU/dq
    dKEdp = [] # dKE/dp
    for pos in qlist:
        pos = torch.tensor([pos], dtype = torch.float32).to(device)
        pos.requires_grad_(True)
        potential = model_dpdt(p.unsqueeze(0), pos.unsqueeze(0))
        dUdq_temp = grad(potential, pos, create_graph = False)[0]
        dUdq.append(dUdq_temp.squeeze().cpu().detach().numpy())
        pot.append(potential.squeeze().cpu().detach().numpy())
        
    for momentum in plist :
        momentum = torch.tensor([momentum], dtype = torch.float32).to(device)
        momentum.requires_grad_(True)
        kin = model_dqdt(momentum.unsqueeze(0) , q.unsqueeze(0))
        dKEdp_temp = grad(kin, momentum, create_graph = False)[0]
        dKEdp.append(dKEdp_temp.squeeze().cpu().detach().numpy())
        kinetic.append(kin.squeeze().cpu().detach().numpy())
        

    plt.plot(qlist, pot ,label = "MD Potential Generator")
    plt.plot(qlist, (qlist ** 2.0 - 1) ** 2.0 + qlist, label = "MD Potential Exact")
    plt.legend(loc = "best")
    plt.ylabel("energy")
    plt.xlabel("q")
    plt.show()
    
    np.save('hamiltonian_q_{}using{}_{}_N{}_{}.npy'.
                format(str(time_step).replace('.',''),
                       str(ground_truth_step).replace('.',''),
                       total_step, 
                       N,
                       hidden_units), np.array((qlist, pot )))
    
    plt.plot(plist, kinetic , label = "MD Kinetic Part")
    plt.plot(plist , (plist ** 2.0 / 2), label = "MD Kinetic Exact")
    plt.legend(loc = "best")
    plt.xlabel("p")
    plt.ylabel("energy")
    plt.show()
    
    np.save('hamiltonian_p_{}using{}_{}_N{}_{}.npy'.
                format(str(time_step).replace('.',''),
                       str(ground_truth_step).replace('.',''),
                       total_step, 
                       N,
                       hidden_units) , np.array((plist, kinetic )))
    
# =============================================================================
#     Get gradient Plot
# =============================================================================
    
    plt.plot(qlist, dUdq, label = "dUdq Predicted ( ML )")
    plt.plot(qlist, (4 * qlist * (qlist ** 2.0 - 1) + 1), label = "dUdq Exact")
    plt.legend(loc = "best")
    plt.ylabel("gradient / dHdq")
    plt.xlabel("q")
    plt.show()
    
    np.save('hamiltonian-grad_q_{}using{}_{}_N{}_{}.npy'.
                format(str(time_step).replace('.',''),
                       str(ground_truth_step).replace('.',''),
                       total_step, 
                       N,
                       hidden_units) , np.array((qlist, dUdq)))
    
    plt.plot(plist, dKEdp, label ="dKEdp predicted (ML)")
    plt.plot(plist, plist, label = "dKEdp Exact")
    plt.legend(loc = 'best')
    plt.ylabel("gradient / dHdp")
    plt.xlabel("p")
    plt.show()
    
    np.save('hamiltonian-grad_p_{}using{}_{}_N{}_{}.npy'.
                format(str(time_step).replace('.',''),
                       str(ground_truth_step).replace('.',''),
                       total_step, 
                       N,
                       hidden_units), np.array((plist, dKEdp)))
    
