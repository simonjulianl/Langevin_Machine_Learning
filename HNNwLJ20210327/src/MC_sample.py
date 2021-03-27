import sys
import os
sys.path.append(os.path.abspath("./parameters"))

from HNN.pair_wise_HNN import pair_wise_HNN
from HNN.models.pair_wise_MLP import pair_wise_MLP
from phase_space import phase_space
from utils.data_io import data_io
from parameters.MC_parameters import MC_parameters
import integrator as integrator
import matplotlib.pyplot as plt
import torch

import numpy as np
import random

# seed setting
seed = MC_parameters.seed
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

text=''

nsamples = MC_parameters.mcstep
nparticle = MC_parameters.nparticle
boxsize = MC_parameters.boxsize
mass = MC_parameters.mass
rho= MC_parameters.rho
temp = MC_parameters.temperature
interval = MC_parameters.interval # take mc step every interval
DISCARD = MC_parameters.DISCARD
new_mcs = MC_parameters.new_mcs
mode = MC_parameters.mode
seed = MC_parameters.seed

print('temp new_mcs nparticle boxsize interval DISCARD iterations iterations - DISCARD mode seed')
print(temp, new_mcs, nparticle, boxsize, interval, DISCARD, MC_parameters.iterations, MC_parameters.num_interval, mode, seed)

uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])
base_dir = uppath(__file__, 2)

init_path = base_dir + '/data/init_config/'

if not os.path.exists(init_path):
                os.makedirs(init_path)

filename = 'nparticle{}_new_nsim_rho{}_T{}_pos_{}_sampled.pt'.format(nparticle, rho, temp, mode)

data_io_obj = data_io(init_path)
phase_space = phase_space.phase_space()
pair_wise_HNN_obj = pair_wise_HNN(pair_wise_MLP())
noMLhamiltonian = super(type(pair_wise_HNN_obj), pair_wise_HNN_obj)

metropolis_mc = integrator.metropolis_mc()
q_hist, U, ACCRatio, spec = metropolis_mc.step(noMLhamiltonian, phase_space)

# take q every interval
q_hist = q_hist[:, 0::interval] # shape is  [new_mcs, mc step, nparticle, DIM]
q_hist = torch.reshape(q_hist, (-1, q_hist.shape[2], q_hist.shape[3])) # shape is  [(new_mcs x mc step), nparticle, DIM]

# momentum sampler
momentum_sampler = integrator.momentum_sampler(q_hist.shape[0])
p_hist = momentum_sampler.momentum_samples()

print('q,p shape', q_hist.shape, p_hist.shape)
qp_list = torch.stack((q_hist,p_hist)) # shape is [(q,p), (new_mcs x mc step), nparticle, DIM]
data_io_obj.write_init_qp(qp_list, filename)


