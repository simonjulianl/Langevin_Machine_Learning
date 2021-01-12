import hamiltonian as hamiltonian
from hamiltonian.pb import pb
from phase_space import phase_space
import integrator as integrator
import matplotlib.pyplot as plt
import torch
import os
import math

#random.seed(43893324) # for train/valide
seed = 23645 # for test

text=''

nsamples = 2
nparticle = 2
DIM = 2
mass = 1
epsilon = 1.
sigma = 1.
rho= 0.1
T = 0.04
dq = 0.1
boxsize = math.sqrt(nparticle/rho)

interval = 40 # take mc step every interval
iterations = 2000
DISCARD = iterations - ( nsamples * interval)

noML_hamiltonian = hamiltonian.hamiltonian()
LJ = hamiltonian.LJ_term(epsilon =1, sigma =1, boxsize=boxsize)
noML_hamiltonian.append(hamiltonian.lennard_jones(LJ , boxsize=boxsize)) #'density': 0.2
noML_hamiltonian.append(hamiltonian.kinetic_energy(mass = 1))

phase_space = phase_space.phase_space()
pb = pb()

state = {
    'seed' : seed,
    'kB' : 1.0, # put as a constant
    'temperature' : T,
    'DIM' : DIM,
    'm' : mass,
    'nsamples' : nsamples, # for train 50000 for test 5000 each temp
    'nparticle' : nparticle,
    'boxsize': boxsize,
    'phase_space': phase_space,
    'pb_q': pb,
    'iterations': iterations,  # for test 44000 #for train 224000 624000 62440
    'DISCARD': DISCARD,  # for test 24000 #for train 24000  622000 62400
    'dq': dq
    }

metropolis_mc = integrator.metropolis_mc(noML_hamiltonian, **state)
q_hist, PE, ACCRatio, spec = metropolis_mc.integrate()

base_library = os.path.abspath('init_config')

# momentum sampler
momentum_sampler = integrator.momentum_sampler(**state)
p_hist = momentum_sampler.momentum_samples()

phase_space = torch.stack((q_hist[0::interval],p_hist))

torch.save(phase_space,base_library+ "/N_particle{}_samples{}_rho{}_T{}_pos_sampled.pt".format(nparticle,q_hist[0::interval].shape[0],rho,T))
