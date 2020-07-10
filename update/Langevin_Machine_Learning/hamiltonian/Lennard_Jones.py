#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 17:21:03 2020

@author: simon
"""
import numpy as np
from .Interaction import Interaction

class Lennard_Jones(Interaction):
    def __init__(self, epsilon : float, sigma : float):
        '''
        Parameters
        ----------
        epsilon : float
            depth of potential well
        sigma : float
            finite distance at which the inter-particle potential is zero
        '''
        try:
            self._epsilon = float(epsilon)
            self._sigma = float(sigma)
            #self._cutoff_r = 2.5 * self._sigma
        except :
            raise Exception('sigma / epsilon rror')

        super().__init__('4 * {0} * (({1}/ q) ** 12.0 - ({1}/q) ** 6.0)'.format(self._epsilon, self._sigma))
        print('Lennard_Jones.py call LJ potential')
        self._name = 'Lennard Jones Potential'
        #since interaction is a function of r or delta q instead of q, we need to modift the data

    # def pos_pbc(self,q_state):
    #
    #     N, particle, DIM = q_state.shape
    #     # Refold positions according to periodic boundary conditions
    #     for i in range(DIM):
    #         period = np.where(pos[:, i] > 0.5)
    #         pos[period, i] = pos[period, i] - 1.0
    #         period = np.where(pos[:, i] < -0.5)
    #         pos[period, i] = pos[period, i] + 1.0


    def energy(self, phase_space, BoxSize = 1,periodicty = False):
        '''
        function to calculate the term directly for truncated lennard jones
        
        Returns
        -------
        term : float 
            Hamiltonian calculated

        '''
        #truncated_potential = 4 * self._epsilon * ((1/2.5) ** 12.0 - (1/2.5) ** 6.0)
        print('Lennard_Jones.py energy phase_space',phase_space)
        q_state = phase_space.get_q()
        term = 0
        print('Lennard_Jones.py energy q_state',q_state.shape)
        N, particle,DIM  = q_state.shape # ADD particle
        print('Lennard_Jones.py energy', q_state)
        for k in range(N):
            for i in range(particle-1) : # loop for every pair of q1,q2
                for j in range(i+1, particle) :
                    q1 = q_state[k,i]
                    print('Lennard_Jones.py energy q1', q1)
                    q2 = q_state[k,j]
                    print('Lennard_Jones.py energy q2', q2)
                    delta_q = q2 - q1 # Reduced LJ units
                    print('Lennard_Jones.py energy dq_x ', delta_q[0])
                    print('Lennard_Jones.py energy dq_y ', delta_q[1])
                    if periodicty : # PBC only
                        for l in range(DIM):
                            if np.abs(delta_q[l]) > 0.5:
                                delta_q[l] = delta_q[l] - np.copysign(1.0, delta_q[l])
                                print('Lennard_Jones.py energy delta_q{} pbc'.format(l), delta_q[l])

                    Rij = BoxSize * delta_q  # scale the box to the real units
                    q = np.sqrt(np.dot(Rij, Rij))
                    print('Lennard_Jones.py energy q', q)
                    #term += eval(self._expression) - truncated_potential
                    print('Lennard_Jones.py energy self._expression', self._expression)
                    term += eval(self._expression)

        return term

    def evaluate_derivative_q(self, phase_space, BoxSize = 1,periodicty = False):
        '''
        Function to calculate dHdq
        
        Returns
        -------
        dHdq: np.array 
            dHdq calculated given the terms of N X DIM 

        '''
        q_state = phase_space.get_q()
        p_state = phase_space.get_p()
        assert q_state.shape == p_state.shape and len(q_state.shape) == 3
        dHdq = np.zeros(q_state.shape) #derivative of separable term in N X DIM matrix
        print('Lennard_Jones.py evaluate_derivative_q q_state',q_state.shape)
        N, particle,DIM  = q_state.shape
        print('Lennard_Jones.py',q_state)
        for k in range(N):
            print('Lennard_Jones.py evaluate_derivative_q dHdp',dHdq)
            for i in range(particle-1) : # loop for every pair of q1,q2
                for j in range(i+1, particle) :
                    q1 = q_state[k,i]
                    print('Lennard_Jones.py q1',q1.shape)
                    print(q1)
                    q2 = q_state[k,j]
                    print('Lennard_Jones.py q2',q2.shape)
                    print(q2)
                    delta_q  = q2 - q1  # Reduced LJ units
                    print('Lennard_Jones.py dq_x ',delta_q[0])
                    print('Lennard_Jones.py dq_y ', delta_q[1])
                    if periodicty : # PBC only
                        for l in range(DIM):
                            print('Lennard_Jones.py delta_q{}'.format(l),delta_q[l])
                            if np.abs(delta_q[l]) > 0.5:
                                delta_q[l] = delta_q[l] - np.copysign(1.0, delta_q[l])
                                print('Lennard_Jones.py delta_q{} pbc'.format(l),delta_q[l])

                    #since dUdr = dUdx x/r
                    Rij = BoxSize * delta_q  # scale the box to the real units
                    q = np.sqrt(np.dot(Rij, Rij))
                    print('Lennard_Jones.py q',q)
                    #if q < self._cutoff_r :
                    print('Lennard_Jones.py derivative_q',self._derivative_q)
                    print('Lennard_Jones.py eval(derivative_q)', eval(self._derivative_q))
                    print(eval(self._derivative_q) * delta_q /q)
                    dHdq[k,i] -= eval(self._derivative_q) * delta_q /q
                    dHdq[k,j] += eval(self._derivative_q) * delta_q /q
                    print('Lennard_Jones.py dHdq[{},{}]'.format(k,i), dHdq[k,i])
                    print('Lennard_Jones.py dHdq[{},{}]'.format(k,j), dHdq[k,j])

        print('Lennard_Jones.py evaluate_derivative_q dHdq', dHdq.shape)
        print('Lennard_Jones.py evaluate_derivative_q dHdq', dHdq)
        return dHdq

