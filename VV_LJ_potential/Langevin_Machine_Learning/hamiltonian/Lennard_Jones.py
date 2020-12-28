#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 17:21:03 2020

@author: simon
"""
import numpy as np
from .Interaction import Interaction

class Lennard_Jones(Interaction):
    def __init__(self, phi, boxsize):
        self.phi = phi
        self.boxsize = boxsize
        self._name = 'Lennard Jones Potential'
        print('Lennard_Jones.py call potentials')

    def dimensionless(self,phase_space):
        q_state = phase_space.get_q()
        p_state = phase_space.get_p()
        q_state = q_state / self.boxsize
        phase_space.set_q(q_state)
        return phase_space

    def energy(self, phase_space, pb): # phase space is real-space-unit
        xi_space = self.dimensionless(phase_space)
        return self.phi.energy(xi_space,pb)

    def evaluate_derivative_q(self, phase_space,pb):
        xi_space = self.dimensionless(phase_space)
        dphidq = self.phi.evaluate_derivative_q(xi_space,pb)
        return dphidq