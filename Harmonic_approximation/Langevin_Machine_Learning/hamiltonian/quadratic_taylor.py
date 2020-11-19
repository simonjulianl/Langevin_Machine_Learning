#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 17:21:03 2020

@author: simon
"""
import numpy as np
from .Interaction import Interaction


class quadratic_taylor(Interaction):
    def __init__(self):
        '''
        Parameters
        ----------
        epsilon : float
            depth of potential well
        sigma : float
            finite distance at which the inter-particle potential is zero
        '''

        self._name = 'quadratic_taylor Potential'
        # since interaction is a function of r or delta q instead of q, we need to modift the data

    def energy(self, xi_space, pb):
        '''
        function to calculate the term directly for truncated lennard jones
        
        Returns
        -------
        term : float 
            Hamiltonian calculated

        '''
        xi_state = xi_space.get_q()
        # p_state = xi_space.get_p()
        term = np.zeros(xi_state.shape[0])
        # N : number of data in batch
        # n_particle : number of particles
        # DIM : dimension of xi
        N, N_particle, DIM = xi_state.shape  # ADD particle
        for z in range(N):
            pb.adjust_reduced(xi_state[z])
            _, q = pb.paired_distance_reduced(xi_state[z])
            # print('Lennard_Jones.py evaluate_xi', self._expression)
            # print('Lennard_Jones.py evaluate_xi eval', eval(self._expression))
            LJ = eval(self._expression)
            # print('Lennard_Jones.py LJ', LJ)
            LJ[~np.isfinite(LJ)] = 0
            term[z] = np.sum(LJ) * 0.5
            # print('Lennard_Jones.py term', term[z])

        term = term * self.parameter_term
        # print('Lennard_Jones.py term', term)
        return term

    def evaluate_derivative_q(self, phase_space, pb):
        '''
        Function to calculate dHdq
        
        Returns
        -------
        dphidxi: np.array 
            dphidxi calculated given the terms of N X DIM 

        '''
        q_state = phase_space.get_q()
        p_state = phase_space.get_p()

        V_0 = np.array([[-2],[1],[3]])
        H = np.array([[3, -2, 4], [-2, 6, 2], [4, 2, 3]])
        q_0 = np.array([[0.1],[0.2],[0.4]])
        #print('quadratic',q_state.shape)

        dphidq = np.zeros(q_state.shape)  # derivative of separable term in N X DIM matrix
        N, N_particle, DIM = q_state.shape
        # print('Lennard_Jones.py evaluate_derivative_q xi_state',xi_state)

        for z in range(N):

            dphidq[z] = V_0 + np.dot(H , (q_state[z] - q_0))


        return dphidq


