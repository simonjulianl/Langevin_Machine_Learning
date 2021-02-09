import sys
import os
sys.path.append(os.path.abspath("./parameters"))

from HNN.pair_wise_HNN import pair_wise_HNN
from HNN.models.pair_wise_MLP import pair_wise_MLP
from phase_space import phase_space
from parameters.MC_paramaters import MC_parameters
import integrator as integrator
import matplotlib.pyplot as plt
import torch
import os

#random.seed(43893324) # for train/valide
import numpy as np
import random
np.random.seed(0)
random.seed(43893324)

text=''

nsamples = MC_parameters.nsamples
nparticle = MC_parameters.nparticle
boxsize = MC_parameters.boxsize
mass = MC_parameters.mass
rho= MC_parameters.rho
temp = MC_parameters.temperature
interval = MC_parameters.interval # take mc step every interval
DISCARD = MC_parameters.DISCARD
new_mcs = MC_parameters.new_mcs

print('new_mcs nparticle boxsize interval DISCARD iterations iterations - DISCARD')
print(new_mcs, nparticle, boxsize, interval, DISCARD, MC_parameters.iterations, MC_parameters.num_interval)

phase_space = phase_space.phase_space()
pair_wise_HNN_obj = pair_wise_HNN(pair_wise_MLP())

noMLhamiltonian = super(type(pair_wise_HNN_obj), pair_wise_HNN_obj)

metropolis_mc = integrator.metropolis_mc()
q_hist, U, ACCRatio, spec = metropolis_mc.step(noMLhamiltonian, phase_space)

base_library = os.path.abspath('init_config')

# text = text +  "{0:.3f}".format(temp) + ' ' + ' '.join(map(str,spec) )+ ' ' + str(ACCRatio)  + '\n'
# plt.title('T={}; AccRatio={:.3f}'.format(temp, ACCRatio),fontsize=15)
# plt.plot(U,'k-')
# plt.xlabel('mcs',fontsize=20)
# plt.ylabel(r'$U_{ij}$',fontsize=20)
# plt.savefig(base_library + '/N_particle{}_samples{}_rho{}_T{}'.format(nparticle, q_hist[0::interval].shape[0], rho, temp) +'.png')

# take q every interval
q_hist = q_hist[:, 0::interval] # shape new_mcs X mc step X nparticle X DIM
q_hist = torch.reshape(q_hist, (-1, q_hist.shape[2], q_hist.shape[3]))
# print(q_hist.shape)

# momentum sampler
momentum_sampler = integrator.momentum_sampler(q_hist.shape[0])
p_hist = momentum_sampler.momentum_samples()
# print(p_hist)

phase_space = torch.stack((q_hist,p_hist))
q, p = phase_space

torch.save(phase_space,base_library+ "/nparticle{}_new_nsim{}_rho{}_T{}_pos_sampled.pt".format(nparticle,new_mcs,rho,temp))
