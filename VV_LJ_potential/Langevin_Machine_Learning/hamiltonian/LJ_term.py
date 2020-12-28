#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 17:21:03 2020

@author: simon
"""
import numpy as np
from .Interaction import Interaction

class LJ_term(Interaction):
    def __init__(self, epsilon : float, sigma : float, boxsize : float, q_adj: float):
        '''
        Parameters
        ----------
        epsilon : float
            depth of potential well
        sigma : float
            finite distance at which the inter-particle potential is zero
        '''
        try:
            self._epsilon  = float(epsilon)
            self._sigma    = float(sigma)
            self._boxsize  = float(boxsize)
            self._q_adj = float(q_adj)

        except :
            raise Exception('sigma / epsilon rror')

        self._name = 'Lennard Jones Potential'

    def energy(self, xi_space, pb):
        '''
        function to calculate the term directly for truncated lennard jones

        Returns
        -------
        term : float
            Hamiltonian calculated

        '''
        xi_state = xi_space.get_q()
        term = np.zeros(xi_state.shape[0])
        # N : number of data in batch
        # n_particle : number of particles
        # DIM : dimension of xi
        N, N_particle, DIM = xi_state.shape  # ADD particle

        a12 = (4 * self._epsilon * pow(self._sigma, 12)) / np.power(self._boxsize, 12)
        a6 = (4 * self._epsilon * pow(self._sigma, 6)) / np.power(self._boxsize, 6)

        for z in range(N):

            _, d = pb.paired_distance_reduced(xi_state[z],self._q_adj)

            s12 = 1 / np.power(d + self._q_adj, 12)
            s12[~np.isfinite(s12)] = 0
            s6 = 1 / np.power(d + self._q_adj, 6)
            s6[~np.isfinite(s6)] = 0

            term[z] = np.sum(a12 * s12 - a6 * s6) * 0.5

        return term

    def evaluate_derivative_q(self,xi_space,pb):

        xi_state = xi_space.get_q()
        p_state  = xi_space.get_p()
        dphidxi = np.zeros(xi_state.shape) #derivative of separable term in N X DIM matrix
        N, N_particle,DIM  = xi_state.shape

        a12 = (4 * self._epsilon * pow(self._sigma, 12)) / np.power(self._boxsize, 13)
        a6 = (4 * self._epsilon * pow(self._sigma, 6)) / np.power(self._boxsize, 7)

        for z in range(N):

            delta_xi, d = pb.paired_distance_reduced(xi_state[z],self._q_adj)

            d = np.expand_dims(d,axis=2)

            s12 = -12*(delta_xi)/np.power(d+self._q_adj,14)
            s12[~np.isfinite(s12)] = 0
            s6  = -6*(delta_xi)/np.power(d+self._q_adj,8)
            s6[~np.isfinite(s6)] = 0
            dphidxi[z] = a12*np.sum(s12,axis=1) - a6*np.sum(s6,axis=1)

        return dphidxi
