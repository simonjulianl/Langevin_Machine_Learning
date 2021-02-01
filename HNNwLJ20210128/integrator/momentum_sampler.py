import torch
import numpy as np

class momentum_sampler:

    def __init__(self, temp, nsamples, nparticle, mass):

        vel = []
        # 'generate': 'maxwell'
        sigma = np.sqrt(temp)  # sqrt(kT/m)

        for i in range(nsamples):
            vx = np.random.normal(0.0, sigma, nparticle)
            vy = np.random.normal(0.0, sigma, nparticle)
            vel_xy = np.stack((vx, vy), axis=-1)
            vel.append(vel_xy)

        self.momentum = torch.tensor(vel) * mass

    def momentum_samples(self):
        return self.momentum

