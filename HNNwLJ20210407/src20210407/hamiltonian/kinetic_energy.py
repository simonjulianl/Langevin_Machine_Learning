#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch

class kinetic_energy:

    '''kinetic_energy class for total kinetic energy of p^2/2m'''

    _obj_count = 0

    def __init__(self, mass = 1):
        '''
        Parameters
        ----------
        mass : int
            mass of the particles, taken to be constant for all
        '''

        kinetic_energy._obj_count += 1
        assert (kinetic_energy._obj_count == 1),type(self).__name__ + " has more than one object"

        self.mass = mass
        print('kinetic_energy.py call kinetic')
        self._name = 'Kinetic Energy'

    def energy(self, phase_space):

        p = phase_space.get_p()
        # shape is [nsamples, nparticle, DIM]

        e = torch.sum( p*p / (2 * self.mass), dim = 1) # sum along nparticle , e shape is [nsamples, DIM]
        e = torch.sum( e, dim = 1) # sum along DIM , e shape is [nsamples]

        return  e

    def evaluate_derivative_q(self, phase_space):

        '''function for make zero to prevent KE from being integrated '''

        return torch.zeros(phase_space.get_q().shape)

    def evaluate_second_derivative_q(self, phase_space):

        nsamples, nparticle, DIM  = phase_space.get_q().shape
        return torch.zeros((nsamples, DIM * nparticle, DIM * nparticle))
