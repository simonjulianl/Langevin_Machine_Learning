#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 17:21:03 2020

@author: simon
"""
import numpy as np
from .Interaction import Interaction

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
        self.parameter_term = (4*self._epsilon)*((self._sigma/self._boxsize)**self._exponent )
        print('Lennard_Jones.py call LJ potential')
        self._name = 'Lennard Jones Potential'
        #since interaction is a function of r or delta q instead of q, we need to modift the data


    def energy(self, xi_space, pb):
        '''
        function to calculate the term directly for truncated lennard jones
        
        Returns
        -------
        term : float 
            Hamiltonian calculated

        '''
        xi_state = xi_space.get_q()
        #p_state = xi_space.get_p()
        term = np.zeros(xi_state.shape[0])
        # N : number of data in batch
        # n_particle : number of particles
        # DIM : dimension of xi
        N,N_particle,DIM  = xi_state.shape # ADD particle
        for z in range(N):
            pb.adjust(xi_state[z])
            _, q = pb.paired_distance(xi_state[z])
            print('Lennard_Jones.py evaluate_xi', self._expression)
            print('Lennard_Jones.py evaluate_xi eval', eval(self._expression))
            LJ = eval(self._expression)
            LJ[~np.isfinite(LJ)] = 0
            term[z] = np.sum(LJ)*0.5
            print('Lennard_Jones.py term', term[z])

        term = term * self.parameter_term
        return term

    def evaluate_derivative_q(self, xi, pb):
        '''
        Function to calculate dHdq
        
        Returns
        -------
        dphidxi: np.array 
            dphidxi calculated given the terms of N X DIM 

        '''
        xi_state = xi_space.get_q()
        p_state  = xi_space.get_p()
        dphidxi = np.zeros(xi_state.shape) #derivative of separable term in N X DIM matrix
        N, N_particle,DIM  = xi_state.shape
        print('Lennard_Jones.py evaluate_derivative_q xi_state',xi_state)

        for z in range(N):
            pb.adjust(xi_state[z])
            delta_xi, q = pb.paired_distance(xi_state[z])
            print('Lennard_Jones.py evaluate_derivative_q derivative_xi', self._derivative_q)
            dphidq = eval(self._derivative_q)
            dphidq[~np.isfinite(dphidq)] = 0
            print('Lennard_Jones.py evaluate_derivative_q dphidq', dphidq)
            dphidxi[z] = np.sum(dphidq,axis=1) * delta_xi / np.sum(q,axis=1)

<<<<<<< HEAD
        dphidx = dphidxi * (self.parameter_term / self._boxsize )
        print('Lennard_Jones.py evaluate_derivative_q dHdq end', dphidxi)
        #print('Lennard_Jones.py evaluate_derivative_q dHdq end', dHdq)
        return dphidx
=======
        dphidq_ = dphidxi * (self.parameter_term / self._boxsize )
        print('Lennard_Jones.py evaluate_derivative_q dHdq end', dphidxi)
        #print('Lennard_Jones.py evaluate_derivative_q dHdq end', dHdq)
        return dphidq_
>>>>>>> a950a1a7822747baf69d59c650ae65659084c919

