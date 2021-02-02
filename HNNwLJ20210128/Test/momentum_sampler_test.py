from HNNwLJ20210128.integrator.momentum_sampler import momentum_sampler
import numpy as np

temp = 0.4
nsamples = 5
nparticle = 2
mass = 1.

np.random.seed(0)
momentum_sampler = momentum_sampler(temp, nsamples, nparticle, mass)
p_list = momentum_sampler.momentum_samples()
print(p_list)
print(p_list.shape)