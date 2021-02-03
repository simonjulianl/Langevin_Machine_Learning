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
mass = MC_parameters.mass
rho= MC_parameters.rho
temp = MC_parameters.temperature
interval = MC_parameters.interval # take mc step every interval
DISCARD = MC_parameters.DISCARD

phase_space = phase_space.phase_space()
pair_wise_HNN_obj = pair_wise_HNN(pair_wise_MLP())

noMLhamiltonian = super(type(pair_wise_HNN_obj), pair_wise_HNN_obj)

metropolis_mc = integrator.metropolis_mc()
q_hist, U, ACCRatio, spec = metropolis_mc.integrate(noMLhamiltonian, phase_space)

base_library = os.path.abspath('init_config')

# text = text +  "{0:.3f}".format(temp) + ' ' + ' '.join(map(str,spec) )+ ' ' + str(ACCRatio)  + '\n'
# plt.title('T={}; AccRatio={:.3f}'.format(temp, ACCRatio),fontsize=15)
# plt.plot(U,'k-')
# plt.xlabel('mcs',fontsize=20)
# plt.ylabel(r'$U_{ij}$',fontsize=20)
# plt.savefig(base_library + '/N_particle{}_samples{}_rho{}_T{}'.format(nparticle, q_hist[0::interval].shape[0], rho, temp) +'.png')

# momentum sampler
momentum_sampler = integrator.momentum_sampler(q_hist[0::interval].shape[0])
p_hist = momentum_sampler.momentum_samples()

phase_space = torch.stack((q_hist[0::interval],p_hist))
q, p = phase_space

torch.save(phase_space,base_library+ "/N_particle{}_samples{}_rho{}_T{}_pos_sampled.pt".format(nparticle,q_hist[0::interval].shape[0],rho,temp))
