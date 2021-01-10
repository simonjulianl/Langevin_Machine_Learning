#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

class LJ_term:
    def __init__(self, epsilon : float, sigma : float, boxsize : float):

        try:
            self._epsilon  = float(epsilon)
            self._sigma    = float(sigma)
            self._boxsize  = float(boxsize)

        except :
            raise Exception('sigma / epsilon rror')

        print('lennard_jones.py call LJ potential')
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

        # N : nsamples
        # n_particle : number of particles
        # DIM : dimension of xi

        N, N_particle, DIM  = xi_state.shape

        a12 = (4 * self._epsilon * pow(self._sigma, 12)) / np.power(self._boxsize, 12)
        a6 = (4 * self._epsilon * pow(self._sigma, 6)) / np.power(self._boxsize, 6)

        for z in range(N):

            _, d = pb.paired_distance_reduced(xi_state[z])

            s12 = 1 /np.power(d,12)
            s12[~np.isfinite(s12)] = 0

            s6  = 1 /np.power(d,6)
            s6[~np.isfinite(s6)] = 0

            term[z] = np.sum(a12* s12 - a6* s6) * 0.5

        return term


    def evaluate_derivative_q(self,xi_space,pb):

        xi_state = xi_space.get_q()
        dphidxi = np.zeros(xi_state.shape) # derivative terms of nsamples
        N, N_particle, DIM  = xi_state.shape

        a12 = (4 * self._epsilon * pow(self._sigma, 12)) / np.power(self._boxsize, 13)
        a6 = (4 * self._epsilon * pow(self._sigma, 6)) / np.power(self._boxsize, 7)

        for z in range(N):

            delta_xi, d = pb.paired_distance_reduced(xi_state[z])
            d = np.expand_dims(d,axis=2)

            s12 = -12*(delta_xi)/np.power(d,14)
            s12[~np.isfinite(s12)] = 0

            s6  = -6*(delta_xi)/np.power(d,8)
            s6[~np.isfinite(s6)] = 0
            dphidxi[z] = a12*np.sum(s12,axis=1) - a6*np.sum(s6,axis=1) # np.sum axis=1 j != k

        return dphidxi

    def evaluate_second_derivative_q(self,xi_space,pb):

        xi_state = xi_space.get_q()
        d2phidxi2_arr = []

        N, N_particle, DIM  = xi_state.shape
        d2phidxi2 = np.zeros((N, N_particle * DIM, N_particle * DIM)) # second derivative terms of nsamples
        d2phidxi_lk = np.zeros((2,2))

        a12 = (4 * self._epsilon * pow(self._sigma, 12)) / np.power(self._boxsize, 14)
        a6 = (4 * self._epsilon * pow(self._sigma, 6)) / np.power(self._boxsize, 8)

        for z in range(N):

            d2phidxi2_ = np.empty((0, N_particle * DIM))

            delta_xi, d = pb.paired_distance_reduced(xi_state[z])
            d = np.expand_dims(d,axis=2)

            s12_same_term = 1. / np.power(d,14)
            s12_same_term[~np.isfinite(s12_same_term)] = 0

            s12_lxkx_lyky = (-14) * np.power(delta_xi,2)/np.power(d,2)
            s12_lxkx_lyky[~np.isfinite(s12_lxkx_lyky)] = 0

            s12_lxky_lykx = 1. /np.power(d,16)
            s12_lxky_lykx[~np.isfinite(s12_lxky_lykx)] = 0

            s6_same_term = 1. / np.power(d,8)
            s6_same_term[~np.isfinite(s6_same_term)] = 0

            s6_lxkx_lyky = (-8) * np.power(delta_xi,2)/np.power(d,2)
            s6_lxkx_lyky[~np.isfinite(s6_lxkx_lyky)] = 0

            s6_lxky_lykx = 1. /np.power(d,10)
            s6_lxky_lykx[~np.isfinite(s6_lxky_lykx)] = 0

            for l in range(N_particle):
                for k in range(N_particle):

                    if l == k:
                        d2phidxi_lxkx = a12 *(-12)* (np.sum(s12_same_term[k] * np.expand_dims(s12_lxkx_lyky[k,:,0],axis=2) + s12_same_term[k],axis=0)) \
                                        - a6 *(-6)* (np.sum(s6_same_term[k] * np.expand_dims(s6_lxkx_lyky[k,:,0],axis=2) + s6_same_term[k],axis=0))

                        d2phidxi_lxky = a12 * (-12)*(-14)*(np.sum(s12_lxky_lykx[k]*np.expand_dims(delta_xi[k,:,0],axis=2)*np.expand_dims(delta_xi[k,:,1],axis=2),axis=0))  \
                                        - a6 * (-6)*(-8)*(np.sum(s6_lxky_lykx[k]*np.expand_dims(delta_xi[k,:,0],axis=2)*np.expand_dims(delta_xi[k,:,1],axis=2),axis=0))

                        d2phidxi_lykx = a12 * (-12)*(-14)*(np.sum(s12_lxky_lykx[k]*np.expand_dims(delta_xi[k,:,0],axis=2)*np.expand_dims(delta_xi[k,:,1],axis=2),axis=0)) \
                                        - a6 * (-6)*(-8)*(np.sum(s6_lxky_lykx[k]*np.expand_dims(delta_xi[k,:,0],axis=2)*np.expand_dims(delta_xi[k,:,1],axis=2),axis=0))

                        d2phidxi_lyky = a12 *(-12)* (np.sum(s12_same_term[k] * np.expand_dims(s12_lxkx_lyky[k,:,1],axis=2) + s12_same_term[k],axis=0)) \
                                        - a6 *(-6)* (np.sum(s6_same_term[k] * np.expand_dims(s6_lxkx_lyky[k,:,1],axis=2) + s6_same_term[k],axis=0))

                        d2phidxi_lk = np.array((d2phidxi_lxkx[0], d2phidxi_lxky[0], d2phidxi_lykx[0], d2phidxi_lyky[0])).reshape(2, 2)

                    if l != k:
                        d2phidxi_lxkx = - a12 *(-12)* s12_same_term[k][l] *( s12_lxkx_lyky[k][l][0] + 1) \
                                        + a6 *(-6)* s6_same_term[k][l] * ( s6_lxkx_lyky[k][l][0] + 1)

                        d2phidxi_lxky = - a12 * (-12)*(-14) * (s12_lxky_lykx[k][l] * delta_xi[k][l][0] * delta_xi[k][l][1]) \
                                        + a6 * (-6)*(-8) * (s6_lxky_lykx[k][l] * delta_xi[k][l][0] * delta_xi[k][l][1])

                        d2phidxi_lykx = - a12 * (-12)*(-14)*(s12_lxky_lykx[k][l] * delta_xi[k][l][0]* delta_xi[k][l][1]) \
                                        + a6 * (-6)*(-8)* (s6_lxky_lykx[k][l] * delta_xi[k][l][0] * delta_xi[k][l][1])

                        d2phidxi_lyky = - a12 *(-12)* s12_same_term[k][l] * ( s12_lxkx_lyky[k][l][1]  + 1) \
                                        + a6 *(-6)* s6_same_term[k][l]  * ( s6_lxkx_lyky[k][l][1] + 1)

                        d2phidxi_lk = np.array((d2phidxi_lxkx[0],d2phidxi_lxky[0],d2phidxi_lykx[0],d2phidxi_lyky[0])).reshape(2,2)

                    d2phidxi2_arr.append(d2phidxi_lk)

                if k == N_particle-1:
                    temp = np.array(d2phidxi2_arr)
                    temp = temp.transpose((1,0,2)).reshape(2, N_particle * DIM )
                    d2phidxi2_arr = []

                d2phidxi2_ = np.append(d2phidxi2_,temp,axis=0)

            d2phidxi2[z] = d2phidxi2_

        return d2phidxi2
