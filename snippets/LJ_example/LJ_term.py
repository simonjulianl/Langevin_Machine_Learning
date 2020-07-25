#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 17:21:03 2020

@author: simon
"""
import numpy as np
from Interaction import Interaction

class LJ_term(Interaction):
    def __init__(self, epsilon : float, sigma : float, exponent : float, boxsize : float):
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
            self._exponent = float(exponent)

        except :
            raise Exception('sigma / epsilon rror')

        super().__init__('1.0 / q ** {0} '.format(self._exponent))
        print('Lennard_Jones.py call LJ potential')
        self._name = 'Lennard Jones Potential'
        #since interaction is a function of r or delta q instead of q, we need to modift the data


    def energy(self, phase_space, pb):
        '''
        function to calculate the term directly for truncated lennard jones
        
        Returns
        -------
        term : float 
            Hamiltonian calculated

        '''
        xi_state = phase_space.get_q()
        term = np.zeros(xi_state.shape[0])
        # N : number of data in batch
        # n_particle : number of particles
        # DIM : dimension of xi
        N,n_particle,DIM  = xi_state.shape # ADD particle
        for k in range(N):
            pb.adjust(xi_state[k])
            _, q = pb.paired_distance(xi_state[k])  # q=dd=[[0, sqrt((dx)^2+(dy)^2)],[sqrt((dx)^2+(dy)^2),0]]
            print('Lennard_Jones.py evaluate_xi', self._expression)   # 1.0 / q ** 6.0
            print('Lennard_Jones.py evaluate_xi eval', eval(self._expression))  #[[inf,eval(LJ)],[eval(LJ),inf]]
            LJ = eval(self._expression)
            LJ[~np.isfinite(LJ)] = 0
            term[k] = np.sum(LJ)*0.5
            print('Lennard_Jones.py term', term[k])

        term = term * (4*self._epsilon)*((self._sigma/self._boxsize)**self._exponent )
        return term

    def evaluate_derivative_q(self, phase_space, pb):
        '''
        Function to calculate dHdq
        
        Returns
        -------
        dphidxi: np.array 
            dphidxi calculated given the terms of N X DIM 

        '''
        xi_state = phase_space.get_q()
        p_state  = phase_space.get_p()
        dphidxi = np.zeros(xi_state.shape) #derivative of separable term in N X DIM matrix
        N, particle,DIM  = xi_state.shape
        print('Lennard_Jones.py evaluate_derivative_q xi_state',xi_state)

        for k in range(N):
            pb.adjust(xi_state[k])
            # delta_xi = [[x2-x1,y2-y1],[x1-x2,y1-y2]]
            # q=dd=[[0, sqrt((dx)^2+(dy)^2)],[sqrt((dx)^2+(dy)^2),0]]
            delta_xi, q = pb.paired_distance(xi_state[k])
            print('Lennard_Jones.py evaluate_derivative_q derivative_xi', self._derivative_q)
            python_derivative_q = eval(self._derivative_q)   # -6.0*q**(-7.0) = -6.0*xi**(-7.0)
            python_derivative_q[~np.isfinite(python_derivative_q)] = 0
            print('Lennard_Jones.py evaluate_derivative_q python_derivative_q', python_derivative_q)  #[[0,eval(derivative_LJ)],[eval(derivative_LJ),0]]
            dphidxi[k] = np.sum(python_derivative_q,axis=1) * delta_xi / np.sum(q,axis=1)  # dphidxi = [[dphi/x1,dph/y1],[dphi/x2,dph/y2]]

        dphidxi = dphidxi * (4 * self._epsilon) * ((self._sigma / self._boxsize) ** self._exponent / self._boxsize )
        print('Lennard_Jones.py evaluate_derivative_q dHdq end', dphidxi)
        #print('Lennard_Jones.py evaluate_derivative_q dHdq end', dHdq)
        return dphidxi

