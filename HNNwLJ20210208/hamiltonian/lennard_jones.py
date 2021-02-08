#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .LJ_term import LJ_term
from MD_paramaters import MD_parameters

class lennard_jones:

    _obj_count = 0

    def __init__(self):

        lennard_jones._obj_count += 1
        assert (lennard_jones._obj_count == 1),type(self).__name__ + " has more than one object"

        self.epsilon = MD_parameters.epsilon
        self.sigma = MD_parameters.sigma
        self.boxsize = MD_parameters.boxsize

        self.phi = LJ_term(self.epsilon, self.sigma, self.boxsize)

        print('lennard_jones.py call potential')
        self._name = 'Lennard Jones Potential'

    def dimensionless(self,phase_space):
        q_state = phase_space.get_q()
        q_state = q_state / self.boxsize
        phase_space.set_q(q_state)
        return phase_space

    def dimensionless_grid(self, grid):
        grid_state = grid / self.boxsize # dimensionless
        return grid_state

    def get_epsilon(self):
        return self.phi._epsilon

    def get_sigma(self):
        return self.phi._sigma

    # data for pair-wise potentials between each grid and particles
    def phi_npixels(self,phase_space, grid):
        grid_state = self.dimensionless_grid(grid)
        xi_space = self.dimensionless(phase_space)
        return self.phi.phi_npixels(xi_space, grid_state)

    def energy(self, phase_space): # phase space is real-space-unit
        xi_space = self.dimensionless(phase_space)
        return self.phi.energy(xi_space)

    def evaluate_derivative_q(self, phase_space):
        xi_space = self.dimensionless(phase_space)
        dphidq = self.phi.evaluate_derivative_q(xi_space)
        return dphidq

    def evaluate_second_derivative_q(self, phase_space):
        xi_space = self.dimensionless(phase_space)
        d2phidq2 = self.phi.evaluate_second_derivative_q(xi_space)
        return d2phidq2
