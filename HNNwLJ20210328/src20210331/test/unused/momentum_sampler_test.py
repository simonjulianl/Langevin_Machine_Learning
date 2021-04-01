# from ..integrator.momentum_sampler import momentum_sampler
from HNNwLJ20210202.integrator.momentum_sampler import momentum_sampler
import numpy as np

if __name__ == '__main__' :

    np.random.seed(0)
    momentum_sampler = momentum_sampler()
    p_list = momentum_sampler.momentum_samples()
    print(p_list)
    print(p_list.shape)