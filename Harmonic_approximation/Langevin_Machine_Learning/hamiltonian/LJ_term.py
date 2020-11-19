#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 17:21:03 2020

@author: simon
"""
import numpy as np
from .Interaction import Interaction

class LJ_term(Interaction):
    def __init__(self, epsilon : float, sigma : float, boxsize : float):
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

        except :
            raise Exception('sigma / epsilon rror')

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
        term = np.zeros(xi_state.shape[0])
        # N : number of data in batch
        # n_particle : number of particles
        # DIM : dimension of xi
        N,N_particle,DIM  = xi_state.shape # ADD particle

        eps = 0
        a12 = (4 * self._epsilon * pow(self._sigma, 12)) / np.power(self._boxsize, 12)
        a6 = (4 * self._epsilon * pow(self._sigma, 6)) / np.power(self._boxsize, 6)

        for z in range(N):

            _, d = pb.paired_distance_reduced(xi_state[z])

            s12 = 1 /np.power(d+eps,12)
            s12[~np.isfinite(s12)] = 0
            s6  = 1 /np.power(d+eps,6)
            s6[~np.isfinite(s6)] = 0

            term[z] = np.sum(a12* s12 - a6* s6) * 0.5

        return term


    def evaluate_derivative_q(self,xi_space,pb):

        xi_state = xi_space.get_q()
        p_state  = xi_space.get_p()
        dphidxi = np.zeros(xi_state.shape) #derivative of separable term in N X DIM matrix
        N, N_particle,DIM  = xi_state.shape

        eps = 0
        a12 = (4 * self._epsilon * pow(self._sigma, 12)) / np.power(self._boxsize, 13)
        a6 = (4 * self._epsilon * pow(self._sigma, 6)) / np.power(self._boxsize, 7)

        for z in range(N):

            delta_xi, d = pb.paired_distance_reduced(xi_state[z])
            d = np.expand_dims(d,axis=2)

            s12 = -12*(delta_xi)/np.power(d+eps,14)
            s12[~np.isfinite(s12)] = 0
            s6  = -6*(delta_xi)/np.power(d+eps,8)
            s6[~np.isfinite(s6)] = 0
            dphidxi[z] = a12*np.sum(s12,axis=1) - a6*np.sum(s6,axis=1)

        return dphidxi

    def evaluate_second_derivative_q(self,xi_space,pb):

        xi_state = xi_space.get_q()
        p_state  = xi_space.get_p()

        d2phidxi2_arr = []

        N, N_particle,DIM  = xi_state.shape
        d2phidxi2_ = np.empty((0,2*N_particle))
        d2phidxi2 = np.zeros((N,2*N_particle,2*N_particle)) #derivative of separable term in N X DIM matrix
        d2phidxi_lk = np.zeros((2,2))

        eps = 0
        a12 = (4 * self._epsilon * pow(self._sigma, 12)) / np.power(self._boxsize, 14)
        a6 = (4 * self._epsilon * pow(self._sigma, 6)) / np.power(self._boxsize, 8)

        for z in range(N):

            delta_xi, d = pb.paired_distance_reduced(xi_state[z])
            d = np.expand_dims(d,axis=2)

            s12_same_term = -12/np.power(d+eps,14)
            s12_same_term[~np.isfinite(s12_same_term)] = 0
            s12_lxkx_lyky = (-12)*(-14) * np.power(delta_xi,2)/np.power(d+eps,2)
            s12_lxkx_lyky[~np.isfinite(s12_lxkx_lyky)] = 0
            s12_lxky_lykx = (-12)*(-14) * delta_xi /np.power(d+eps,16)
            s12_lxky_lykx[~np.isfinite(s12_lxky_lykx)] = 0

            s6_same_term  = -6/np.power(d+eps,8)
            s6_same_term[~np.isfinite(s6_same_term)] = 0
            s6_lxkx_lyky  = (-6)*(-8)*np.power(delta_xi,2)/np.power(d+eps,2)
            s6_lxkx_lyky[~np.isfinite(s6_lxkx_lyky)] = 0
            s6_lxky_lykx = (-6)*(-8)* delta_xi /np.power(d+eps,10)
            s6_lxky_lykx[~np.isfinite(s6_lxky_lykx)] = 0

            for l in range(N_particle):
                print('l',l)
                for k in range(N_particle):
                    print('k', k)
                    if l == k:
                        d2phidxi_lxkx = a12 * np.sum(s12_same_term[k],axis=0) * ( np.sum(s12_lxkx_lyky[k],axis=0)[0] + 1) \
                                        - a6 * np.sum(s6_same_term[k],axis=0) *( np.sum(s6_lxkx_lyky[k],axis=0)[0] + 1)

                        d2phidxi_lxky = a12 * (np.sum(s12_lxkx_lyky[k],axis=0)[0]*np.sum(s12_lxkx_lyky[k],axis=0)[1]) \
                                        - a6 * (np.sum(s6_lxkx_lyky[k],axis=0)[0]*np.sum(s6_lxkx_lyky[k],axis=0)[1])

                        d2phidxi_lykx = a12 * (np.sum(s12_lxkx_lyky[k],axis=0)[0]*np.sum(s12_lxkx_lyky[k],axis=0)[1]) \
                                        - a6 * (np.sum(s6_lxkx_lyky[k],axis=0)[0]*np.sum(s6_lxkx_lyky[k],axis=0)[1])

                        d2phidxi_lyky = a12 * np.sum(s12_same_term[k],axis=0) * ( np.sum(s12_lxkx_lyky[k],axis=0)[1] + 1) \
                                        - a6 * np.sum(s6_same_term[k],axis=0) *( np.sum(s6_lxkx_lyky[k],axis=0)[1] + 1)

                        d2phidxi_lk = np.array((d2phidxi_lxkx[0], d2phidxi_lxky, d2phidxi_lykx, d2phidxi_lyky[0])).reshape(2, 2)
                        print(d2phidxi_lk)

                    if l != k:
                        d2phidxi_lxkx = - a12 * s12_same_term[k][l] *( s12_lxkx_lyky[k][l][0] + 1) \
                                        + a6 * s6_same_term[k][l] * ( s6_lxkx_lyky[k][l][0] + 1)

                        d2phidxi_lxky = - a12 * (s12_lxkx_lyky[k][l][0]* s12_lxkx_lyky[k][l][1]) \
                                        + a6 * (s6_lxkx_lyky[k][l][0]*s6_lxkx_lyky[k][l][1])

                        d2phidxi_lykx = - a12 * (s12_lxkx_lyky[k][l][0]* s12_lxkx_lyky[k][l][1]) \
                                        + a6 * (s6_lxkx_lyky[k][l][0]*s6_lxkx_lyky[k][l][1])

                        d2phidxi_lyky = - a12 * s12_same_term[k][l] * ( s12_lxkx_lyky[k][l][1]  + 1) \
                                        + a6 * s6_same_term[k][l]  * ( s6_lxkx_lyky[k][l][1] + 1)

                        d2phidxi_lk = np.array((d2phidxi_lxkx[0],d2phidxi_lxky,d2phidxi_lykx,d2phidxi_lyky[0])).reshape(2,2)
                        print(d2phidxi_lk)

                    d2phidxi2_arr.append(d2phidxi_lk)

                if k == N_particle-1:
                    temp = np.array(d2phidxi2_arr)
                    temp = temp.transpose((1,0,2)).reshape(2,2*N_particle)
                    d2phidxi2_arr = []

                d2phidxi2_ = np.append(d2phidxi2_,temp,axis=0)

            d2phidxi2[z] = d2phidxi2_
        print(d2phidxi2)

        #print('quadratic_taylor.py evaluate_derivative_q dHdq_ end', dphidxi)
        return d2phidxi2