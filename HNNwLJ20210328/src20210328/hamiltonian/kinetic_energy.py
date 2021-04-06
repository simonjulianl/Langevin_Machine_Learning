#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch

class kinetic_energy:

    '''kinetic_energy class for total kinetic energy of p^2/2m'''

    _obj_count = 0

    def __init__(self, mass : float):
        '''
        Parameters
        ----------
        mass : float
            mass of the particles, taken to be constant for all
        '''

        kinetic_energy._obj_count += 1
        assert (kinetic_energy._obj_count == 1),type(self).__name__ + " has more than one object"

        self.mass = mass
        print('kinetic_energy.py call kinetic')
        self._name = 'Kinetic Energy'

    def energy(self, phase_space): # phase space is real-space-unit

        p = phase_space.get_p()

        e_element = torch.sum( p*p / (2 * self.mass), dim = 1) # x, y , shape is [nparticle, DIM]
        e = torch.sum( e_element, dim = 1) # sum each element x, y , shape is [nparticle]

        return  e

    def evaluate_derivative_q(self, phase_space):

        '''function for make zero to prevent KE from being integrated '''

        return torch.zeros(phase_space.get_q().shape)

    def evaluate_second_derivative_q(self, phase_space):

        nsamples, nparticle, DIM  = phase_space.get_q().shape
        return torch.zeros((nsamples, DIM * nparticle, DIM * nparticle))