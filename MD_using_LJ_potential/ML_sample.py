import Langevin_Machine_Learning.hamiltonian as Hamiltonian
import Langevin_Machine_Learning.Integrator as Integrator
import Langevin_Machine_Learning.Integrator.methods as methods
import Langevin_Machine_Learning.pair_wise_HNN as pair_wise_HNN
import Langevin_Machine_Learning.pair_wise_HNN.models as models
from Langevin_Machine_Learning.pair_wise_HNN.loss import qp_MSE_loss

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
DIM = 2
batch_size = 1
nsamples = 1 # not use when ML

print("iterations",iterations)
print("lr",lr)
print("batch size",batch_size)

# noML_hamiltonian
noML_hamiltonian = Hamiltonian.Hamiltonian()
LJ = Hamiltonian.LJ_term(epsilon =1, sigma =1, boxsize=np.sqrt(N_particle/rho))
noML_hamiltonian.append(Hamiltonian.Lennard_Jones(LJ, boxsize=np.sqrt(N_particle/rho)))
noML_hamiltonian.append(Hamiltonian.kinetic_energy(mass = 1))

configuration = {
    'kB' : 1.0, # put as a constant
    'DIM' : 2,
    'rho':rho,
    'BoxSize': np.sqrt(N_particle/rho),
    'particle' : N_particle, #Num of particle
    'N': nsamples,  # Total number of samples for train 50000 for test 5000
    'Temperature': T,
    'm' : 1,
    'hamiltonian' : noML_hamiltonian
    }

integration_setting = {
    'iterations' : iterations,
    'gamma' : 0, # turn off Langevin heat bath
    'tau' : tau, # time step for label
    'integrator_method' : methods.linear_velocity_verlet
    }

configuration.update(integration_setting)
configuration.update(integration_setting)
print('==')
print(noML_hamiltonian)

# noML + ML
MLP = models.pair_wise_MLP.pair_wise_MLP(5,20)
pairwise_HNN = pair_wise_HNN.pair_wise_HNN(MLP, **configuration)

loss = qp_MSE_loss

NN_trainer_setting = {
    'optim': optim.Adam(MLP.parameters(), lr=lr),
    'model' : MLP,
    'loss' : loss,
    'epoch' : 3,
    'batch_size' : batch_size,
    'general_hamiltonian' : pairwise_HNN
    }


configuration.update(NN_trainer_setting)
print(configuration)
MD_learner = pair_wise_HNN.MD_learner(**configuration)
MD_learner.train()


