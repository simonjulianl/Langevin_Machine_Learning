import torch
import numpy as np
from parameters.MD_paramaters import MD_parameters

class momentum_sampler:

    def __init__(self, interval_nsamples):

        self.vel = np.zeros((interval_nsamples, MD_parameters.nparticle, MD_parameters.DIM))

    def momentum_samples(self):
        # 'generate': 'maxwell'
        sigma = np.sqrt( MD_parameters.temp )  # sqrt(kT/m)
        # vx = np.random.normal(0.0, sigma, nparticle)
        # vy = np.random.normal(0.0, sigma, nparticle)
        # vel_xy = np.stack((vx, vy), axis=-1)
        self.vel = np.random.normal(0, 1, (self.vel.shape[0], self.vel.shape[1],self.vel.shape[2])) * sigma # make sure shape correct
        momentum = torch.tensor(self.vel) * MD_parameters.mass

        return momentum

