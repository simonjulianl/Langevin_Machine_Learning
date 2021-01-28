import torch
import numpy as np

class momentum_sampler:

    def __init__(self, **state):

        vel = []
        # 'generate': 'maxwell'
        sigma = np.sqrt(state['temperature'])  # sqrt(kT/m)

        for i in range(state['nsamples']):
            vx = np.random.normal(0.0, sigma, state['nparticle'])
            vy = np.random.normal(0.0, sigma, state['nparticle'])
            vel_xy = np.stack((vx, vy), axis=-1)
            vel.append(vel_xy)

        self.momentum = torch.tensor(vel) * state['m']

    def momentum_samples(self):
        return self.momentum

