import torch
import numpy as np
from parameters.MC_parameters import MC_parameters

class momentum_sampler:

    _obj_count = 0

    def __init__(self, interval_nsamples):

        momentum_sampler._obj_count += 1
        assert (momentum_sampler._obj_count == 1),type(self).__name__ + " has more than one object"

        self.vel = np.zeros((interval_nsamples, MC_parameters.nparticle, MC_parameters.DIM))

    def momentum_samples(self):
        # 'generate': 'maxwell'
        sigma = np.sqrt( MC_parameters.temperature )  # sqrt(kT/m)
        # vx = np.random.normal(0.0, sigma, nparticle)
        # vy = np.random.normal(0.0, sigma, nparticle)
        # vel_xy = np.stack((vx, vy), axis=-1)
        self.vel = np.random.normal(0, 1, (self.vel.shape)) * sigma # make sure shape correct
        momentum = torch.tensor(self.vel) * MC_parameters.mass

        return momentum

