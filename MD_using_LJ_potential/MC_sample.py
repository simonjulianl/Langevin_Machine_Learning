import Langevin_Machine_Learning.hamiltonian as Hamiltonian
import Langevin_Machine_Learning.Integrator as Integrator
import Langevin_Machine_Learning.utils as confStat
import Langevin_Machine_Learning.phase_space as phase_space
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import random

#random.seed(43893324) # for train/valide
random.seed(23645) # for test

text=''
N_particle = 3
nsamples = 1
rho= 0.1
T = 0.04
dq = 0.1

print("N_particle",N_particle)
print("rho:",rho)
print("T:",T)
print("dq",dq)

energy = Hamiltonian.Hamiltonian()
LJ = Hamiltonian.LJ_term(epsilon =1, sigma =1, boxsize=np.sqrt(N_particle/rho))
energy.append(Hamiltonian.Lennard_Jones(LJ , boxsize=np.sqrt(N_particle/rho))) #'density': 0.2
energy.append(Hamiltonian.kinetic_energy(mass = 1))

configuration = {
    'kB' : 1.0, # put as a constant
    'Temperature' : T,
    'DIM' : 2,
    'm' : 1,
    'particle' : N_particle,
    'N' : nsamples, # for train 50000 for test 5000 each temp
    'BoxSize': np.sqrt(N_particle/rho),
    'hamiltonian' : energy,
    }

integration_setting = {
    'iterations' : 62440,  #for test 44000 #for train 224000 624000 62440
    'DISCARD' :  62400,  #for test 24000 #for train 24000  622000 62400
    'dq' : dq,
    }

configuration.update(integration_setting) # combine the 2 dictionaries

MSMC_integrator = Integrator.MCMC(**configuration)
q_hist, PE, ACCRatio, spec = MSMC_integrator.integrate()

base_library = os.path.abspath('Langevin_Machine_Learning/init_config')

# momentum sampler
Momentum_sampler = Integrator.momentum_sampler(**configuration)
p_hist = Momentum_sampler.integrate()

interval = 40

phase_space = np.array((q_hist[0::interval],p_hist))
np.save(base_library+ "/N_particle{}_samples{}_rho{}_T{}_pos_sampled.npy".format(N_particle,q_hist[0::interval].shape[0],rho,T),phase_space)

