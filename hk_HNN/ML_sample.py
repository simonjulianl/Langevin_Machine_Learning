from Langevin_Machine_Learning.pair_wise_HNN.loss import qp_MSE_loss
import Langevin_Machine_Learning.pair_wise_HNN as Pair_wise_HNN
import Langevin_Machine_Learning.Integrator as Integrator
import Langevin_Machine_Learning.hamiltonian as noML_hamiltonian
import Langevin_Machine_Learning.Integrator.methods as methods
import Langevin_Machine_Learning.pair_wise_HNN as pair_wise_HNN
import torch
import torch.optim as optim
import numpy as np
import argparse
import os

N_particle = 3
rho= 0.1
T = 0.04
iterations= 10
tau = 0.01
lr= 0.01
#batch_size= N_particle * (N_particle-1)
batch_size = 1
nsamples = 1 # not use when ML

print("iterations",iterations)
print("lr",lr)
print("batch size",batch_size)

# noML_hamiltonian
noML_hamiltonian = noML_hamiltonian.hamiltonian()
energy = pairwise_HNN(noML_hamiltonian,pairwise_MLP())
LJ = Hamiltonian.LJ_term(epsilon =1, sigma =1, boxsize=np.sqrt(N_particle/rho))
energy.append(Hamiltonian.Lennard_Jones(LJ, boxsize=np.sqrt(N_particle/rho)))
energy.append(Hamiltonian.kinetic_energy(mass = 1))

configuration = {
    'kB' : 1.0, # put as a constant
    'DIM' : 2,
    'rho':rho,
    'BoxSize': np.sqrt(N_particle/rho),
    'particle' : N_particle, #Num of particle
    'N': nsamples,  # Total number of samples for train 50000 for test 5000
    'Temperature': T,
    'm' : 1,
    'hamiltonian' : energy,
    'init_config' : np.load('./Langevin_Machine_Learning/init_config/N_particle3_samples1_rho0.1_T0.04_pos_sampled.npy')
    }

integration_setting = {
    'iterations' : iterations,
    'gamma' : 0, # turn off Langevin heat bath
    'tau' : tau, # time step for label
    'integrator_method' : methods.linear_velocity_verlet
    }

#model = Pair_wise_HNN.pair_wise_MLP(batch_size*configuration['DIM'] + 1, 20) # qx, qy, px, py, tau
model = Pair_wise_HNN.pair_wise_MLP((N_particle-1)*configuration['DIM'] + 1, 20) # qx, qy, px, py, tau

loss = qp_MSE_loss

NN_trainer_setting = {
    'optim' : optim.Adam(model.parameters(), lr = lr),
    'model' : model,
    'loss' : loss,
    'epoch' : 3,
    'batch_size' : batch_size,
    'ML_integrator_method' : methods.pair_wise_linear_velocity_verlet
    }

configuration.update(integration_setting)
configuration.update(integration_setting)
configuration.update(NN_trainer_setting)

HNN = Pair_wise_HNN.HNN_trainer(**configuration)
HNN.train()


