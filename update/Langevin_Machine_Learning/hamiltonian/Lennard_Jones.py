#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 17:21:03 2020

@author: simon
"""
import numpy as np
from .Interaction import Interaction

class Lennard_Jones(Interaction):
    def __init__(self, phi06, phi12):
        self.phi06 = phi06
        self.phi12 = phi12
        self._name = 'Lennard Jones Potential'
        #super().__init__('1.0 / q ** {0} '.format(self._exponent))

    def energy(self, phase_space, pb): # phase space is real-space-unit
        xi_space = self.dimensionless(phase_space)
        return self.phi06.energy(xi_space,pb) + self.phi12.energy(xi_space,pb)

    def evaluate_derivative_q(self, phase_space,pb):
        xi_space = self.dimensionless(phase_space)
        dphidq = self.phi06.evaluate_derivative_q(xi_space,pb)+
self.phi12.evaluate_derivative_q(xi_space,pb)
        return dphidq


