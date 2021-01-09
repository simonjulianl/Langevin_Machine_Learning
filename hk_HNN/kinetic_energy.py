#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

class kinetic_energy:
    '''Implemented class for total kinetic energy of p^2/2m'''
    def __init__(self, mass : float):
        '''
        Parameters
        ----------
        mass : float
            mass of the particles, taken to be constant for all
        '''
        try : 
            str_mass = str(float(mass))
        except : 
            raise Exception('Mass is not a float / error in mass')
            
        self.mass = mass
        print('kinetic_energy.py call kinetic')
        self._name = 'Kinetic Energy'

    def energy(self, phase_space, pb): # phase space is real-space-unit
        #p = phase_space.get_p()
        #e = np.sum(p*p / (2*self.mass))
        return np.zeros([]) # make zero for Monte-carlo

    def evaluate_derivative_q(self, phase_space,pb):
        return np.zeros(phase_space.get_q().shape)

    def evaluate_second_derivative_q(self, phase_space,pb):
        N, N_particle, DIM  = phase_space.get_q().shape
        return np.zeros((N, DIM * N_particle, DIM * N_particle))
