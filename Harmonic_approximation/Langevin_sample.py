import Langevin_Machine_Learning.hamiltonian as Hamiltonian
import Langevin_Machine_Learning.Integrator as Integrator
import Langevin_Machine_Learning.Integrator.methods as methods
import Langevin_Machine_Learning.utils as confStat # configuration statistics
import numpy as np
import argparse
import time

N_particle = 4
rho = 0.1
T = 0.04
nsamples = 5
iterations = 10
tau = 0.1
gamma = 0

print("N_particle",N_particle)
print("rho:",rho)
print("T:",T)
print("N_samples",nsamples)
print("iterations",iterations)
print("tau",tau)
print("gamma",gamma)

energy = Hamiltonian.Hamiltonian()
LJ = Hamiltonian.LJ_term(epsilon =1, sigma =1, boxsize=np.sqrt(N_particle/rho))
energy.append(Hamiltonian.Lennard_Jones(LJ, boxsize=np.sqrt(N_particle/rho)))
energy.append(Hamiltonian.kinetic_energy(mass = 1))

configuration = {
    'kB' : 1.0, # put as a constant
    'Temperature' : T, # desired temperature for NVE Ensemble
    'DIM' : 2,
    'm' : 1,
    'particle' : N_particle,
    'N' : nsamples,   # Total number of samples for train 50000 for test 5000
    'BoxSize': np.sqrt(N_particle/rho),  #'density' =particle/volume : 0.1 ; Boxsize : sqrt(particle/density)
    'hamiltonian' : energy
    }

integration_setting = {
    'iterations' : iterations,
    'gamma' : gamma, # gamma 0 turns off the Langevin heat bath, setting it to NVE Ensemble
    'tau' : tau,
    'integrator_method': methods.harmonic_velocity_verlet
    }

#MD simulation
configuration.update(integration_setting)
MD_integrator = Integrator.harmonic_integrator(**configuration)

##########################################################################
#load for initial condition for each temperature
#change filename when test
#change filename when train
# HK: samples -> nsamples
initial_q_hist, initial_p_hist = MD_integrator.set_phase_space(nsamples = nsamples) # command out when save a file
#update configuration after loading
configuration = MD_integrator.get_configuration()
print('Run MD simulation')
start = time.time()
q_hist_, p_hist_ = MD_integrator.integrate()
end = time.time()
print('time',end-start)

# initial_q_hist = np.expand_dims(initial_q_hist, axis=0)
# initial_p_hist = np.expand_dims(initial_p_hist, axis=0)
# q_hist = np.concatenate((initial_q_hist, q_hist_), axis=0)
# p_hist = np.concatenate((initial_p_hist, p_hist_), axis=0)
#
# ##to save the current phase space to continue as a checkpoint
# MD_integrator.save_phase_space(q_hist, p_hist,'/SHO_N{}_T{}_ts{}_iter{}_vv_gm{}_eps{}_{}sampled.npy'.format(N_particle,T,ts,iterations,gamma,q_adj,samples)) # by default, it is in init file
#

