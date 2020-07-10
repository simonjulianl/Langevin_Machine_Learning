#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 12:47:47 2020

@author: simon
"""

from .Interaction import Interaction

class kinetic_energy(Interaction):
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
            
        super().__init__('p**2.0 / (2 * {})'.format(str_mass))
        print('kinetic_energy.py call kinetic')
        self._name = 'Kinetic Energy'
