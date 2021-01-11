import hamiltonian as NoML_hamiltonian
from LJ_term import LJ_term
from lennard_jones import lennard_jones
from kinetic_energy import kinetic_energy


import torch
import torch.optim as optim
import numpy as np
import argparse
import os

nsample = 1
N_particle = 2
DIM = 2
mass = 1
epsilon = 1.
sigma = 1.
boxsize = 6.  # np.sqrt(N_particle/rho)
tau = 0.01
iterations = 10
n_input = 5
n_hidden = 5
lr = 0.01
nepochs = 1


print("iterations",iterations)
print("lr",lr)

# noML_hamiltonian
noML_hamiltonian = NoML_hamiltonian.hamiltonian()
LJ = LJ_term(epsilon = 1, sigma = 1, boxsize = boxsize)
noML_hamiltonian.append(lennard_jones(LJ, boxsize = boxsize))
noML_hamiltonian.append(kinetic_energy(mass = 1))

energy = pairwise_HNN(noML_hamiltonian, pairwise_MLP)

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
    'ML_integrator_method' : methods.linear_velocity_verlet
    }

configuration.update(integration_setting)
configuration.update(integration_setting)
configuration.update(NN_trainer_setting)

HNN = Pair_wise_HNN.HNN_trainer(**configuration)
HNN.train()


