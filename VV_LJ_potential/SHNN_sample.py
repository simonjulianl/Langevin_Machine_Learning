from Langevin_Machine_Learning.HNN.loss import qp_MSE_loss  
import Langevin_Machine_Learning.HNN as HNN
import Langevin_Machine_Learning.hamiltonian as Hamiltonian
import Langevin_Machine_Learning.Integrator.methods as methods
import torch
import torch.optim as optim 
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser('ML simulation')
args = parser.add_argument('--particle', type= int, default = 2, help="particle")
args = parser.add_argument('--rho', type= float, default = 0.1, help="rho")
args = parser.add_argument('--iterations', type= int, default = 1000, help="iterations")
args = parser.add_argument('--lr', type= float, default = 0.1, help="lr")
args = parser.add_argument('--batch', type= int, default = 128, help="batch")
args = parser.parse_args()

N_particle = 32
rho= 0.1
iterations= 1000
lr= 3e-3
batch_size= 128

print("iterations",iterations)
print("lr",lr)
print("batch size",batch_size)

energy = Hamiltonian.Hamiltonian()
LJ = Hamiltonian.LJ_term(epsilon =1, sigma =1, boxsize=np.sqrt(N_particle/rho), q_adj=q_adj)
energy.append(Hamiltonian.Lennard_Jones(LJ , boxsize=np.sqrt(N_particle/rho))) #'density': 0.2
energy.append(Hamiltonian.kinetic_energy(mass = 1))

configuration = {
    'kB' : 1.0, # put as a constant
    'DIM' : 2,
    'rho':rho,
    'BoxSize': np.sqrt(N_particle/rho),
    'particle' : 2, #Num of particle
    'Temperature': 1,
    'm' : 1,
    'hamiltonian' : energy,
    }

#Generate dataset ground truth, turn off Langevin drag coefficient
integration_setting = {
    'iterations' : iterations,
    'DumpFreq' : 1,
    'gamma' : 0, # turn off Langevin heat bath
    'time_step' : 0.001,
    'integrator_method' : methods.velocity_verlet
    }

model = HNN.MLP2H_Separable_Hamil_VV(2*configuration['particle']*configuration['DIM'], 20)
loss = qp_MSE_loss

NN_trainer_setting = {
    'optim' : optim.Adam(model.parameters(), lr = lr),
    'model' : model,
    'loss' : loss,
    'epoch' : 2000,
    'batch_size' : batch_size
    }

#[0.01, 0.04, 0.08, 0.12 ,0.16, 0.2, 0.24, 0.28, 0.32, 0.36, 0.4, 0.44, 0.48, 0.52, 0.56, 0.6, 0.64, 0.68, 0.72, 0.76],
Dataset_setting = {
    'Temperature_List' : [0.01, 0.04, 0.08, 0.12 ,0.16, 0.2, 0.24, 0.28, 0.32, 0.36, 0.4, 0.44, 0.48, 0.52, 0.56, 0.6, 0.64, 0.68, 0.72, 0.76],
    'sample' : 20000, # samples per temperature
    }

configuration.update(integration_setting)
configuration.update(NN_trainer_setting)
configuration.update(Dataset_setting)

print(configuration)
SHNN = HNN.SHNN_trainer(level = 1, folder_name = 'VV_N_particle_{}_split_82_Hidden20_Batch{}_lr{}'.format(N_particle,batch_size,lr), **configuration)
SHNN.train()
