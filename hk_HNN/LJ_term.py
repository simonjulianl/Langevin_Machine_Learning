#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch

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
        term = torch.zeros(xi_state.shape[0])

        # N : nsamples
        # n_particle : number of particles
        # DIM : dimension of xi

        N, N_particle, DIM  = xi_state.shape

        a12 = (4 * self._epsilon * pow(self._sigma, 12)) / pow(self._boxsize, 12)
        a6 = (4 * self._epsilon * pow(self._sigma, 6)) / pow(self._boxsize, 6)

        for z in range(N):

            _, d = pb.paired_distance_reduced(xi_state[z],N_particle,DIM)

            print('d',d)
            s12 = 1 / pow(d,12)
            print('s12',s12)

            s6  = 1 / pow(d,6)

            term[z] = torch.sum(a12* s12 - a6* s6) * 0.5

        return term


    def evaluate_derivative_q(self,xi_space,pb):

        xi_state = xi_space.get_q()
        dphidxi = torch.zeros(xi_state.shape) # derivative terms of nsamples
        N, N_particle, DIM  = xi_state.shape

        a12 = (4 * self._epsilon * pow(self._sigma, 12)) / pow(self._boxsize, 13)
        a6 = (4 * self._epsilon * pow(self._sigma, 6)) / pow(self._boxsize, 7)

        for z in range(N):

            delta_xi, d = pb.paired_distance_reduced(xi_state[z],N_particle,DIM)
            d = torch.unsqueeze(d,dim =2)
            # print('d expand',d)
            # print(delta_xi.shape)

            # print('delta_xi',delta_xi)
            # print('d',d.shape)

            s12 = -12 * (delta_xi) / pow(d,14)
            # print('s12',s12)

            s6  = -6 * (delta_xi) / pow(d,8)

            dphidxi[z] = a12*torch.sum(s12,dim=1) - a6*torch.sum(s6,dim=1) # np.sum axis=1 j != k
            # print('dH/dq',dphidxi[z])

        return dphidxi

    def evaluate_second_derivative_q(self,xi_space,pb):

        xi_state = xi_space.get_q()
        d2phidxi2_append = []

        N, N_particle, DIM  = xi_state.shape
        d2phidxi2 = torch.zeros((N, N_particle * DIM, N_particle * DIM)) # second derivative terms of nsamples
        d2phidxi_lk = torch.zeros((2,2))

        a12 = (4 * self._epsilon * pow(self._sigma, 12)) / pow(self._boxsize, 14)
        a6 = (4 * self._epsilon * pow(self._sigma, 6)) / pow(self._boxsize, 8)

        for z in range(N):

            d2phidxi2_ = torch.empty((0, N_particle * DIM))

            delta_xi, d = pb.paired_distance_reduced(xi_state[z],N_particle,DIM)
            d = torch.unsqueeze(d,dim=2)

            s12_same_term = 1. / pow(d,14)
            # print('s12_same_term',s12_same_term)
            # print('s12_same_term', s12_same_term.shape)
            # s12_same_term[torch.isinf(s12_same_term)] = 0
            #print('s12_same_term',s12_same_term)

            s12_lxkx_lyky = (-14) * torch.pow(delta_xi,2) / torch.pow(d,2)
            # print('s12_lxkx_lyky',s12_lxkx_lyky)
            # print('s12_lxkx_lyky', s12_lxkx_lyky.shape)
            # s12_lxkx_lyky[torch.isnan(s12_lxkx_lyky)] = 0

            s12_lxky_lykx = 1. / pow(d,16)
            # s12_lxky_lykx[torch.isinf(s12_lxky_lykx)] = 0
            # print('s12_lxky_lykx',s12_lxky_lykx)
            # print('s12_lxky_lykx', s12_lxky_lykx.shape)

            s6_same_term = 1. / pow(d,8)
            # s6_same_term[torch.isinf(s6_same_term)] = 0

            s6_lxkx_lyky = (-8) * torch.pow(delta_xi,2) / torch.pow(d,2)
            # s6_lxkx_lyky[torch.isnan(s6_lxkx_lyky)] = 0

            s6_lxky_lykx = 1. / pow(d,10)
            # s6_lxky_lykx[torch.isinf(s6_lxky_lykx)] = 0

            for l in range(N_particle):
                j = 0
                for k in range(N_particle):

                    if l == k:
                        d2phidxi_lxkx = a12 *(-12)* (torch.sum(s12_same_term[k] * torch.unsqueeze(s12_lxkx_lyky[k,:,0],dim=-1) + s12_same_term[k],dim=0)) \
                                        - a6 *(-6)* (torch.sum(s6_same_term[k] * torch.unsqueeze(s6_lxkx_lyky[k,:,0],dim=-1) + s6_same_term[k],dim=0))

                        d2phidxi_lxky = a12 * (-12)*(-14)*(torch.sum(s12_lxky_lykx[k]*torch.unsqueeze(delta_xi[k,:,0],dim=-1)*torch.unsqueeze(delta_xi[k,:,1],dim=-1),dim=0))  \
                                        - a6 * (-6)*(-8)*(torch.sum(s6_lxky_lykx[k]*torch.unsqueeze(delta_xi[k,:,0],dim=-1)*torch.unsqueeze(delta_xi[k,:,1],dim=-1),dim=0))

                        d2phidxi_lykx = a12 * (-12)*(-14)*(torch.sum(s12_lxky_lykx[k]*torch.unsqueeze(delta_xi[k,:,0],dim=-1)*torch.unsqueeze(delta_xi[k,:,1],dim=-1),dim=0)) \
                                        - a6 * (-6)*(-8)*(torch.sum(s6_lxky_lykx[k]*torch.unsqueeze(delta_xi[k,:,0],dim=-1)*torch.unsqueeze(delta_xi[k,:,1],dim=-1),dim=0))

                        d2phidxi_lyky = a12 *(-12)* (torch.sum(s12_same_term[k] * torch.unsqueeze(s12_lxkx_lyky[k,:,1],dim=-1) + s12_same_term[k],dim=0)) \
                                        - a6 *(-6)* (torch.sum(s6_same_term[k] * torch.unsqueeze(s6_lxkx_lyky[k,:,1],dim=-1) + s6_same_term[k],dim=0))

                        d2phidxi_lk = torch.tensor((d2phidxi_lxkx[0], d2phidxi_lxky[0], d2phidxi_lykx[0], d2phidxi_lyky[0])).reshape(2, 2)
                        print('l=k d2phidxi_lk',d2phidxi_lk)

                    if l != k:
                        print('j',j)

                        d2phidxi_lxkx = - a12 *(-12)* s12_same_term[l][j] *( s12_lxkx_lyky[l][j][0] + 1) \
                                        + a6 *(-6)* s6_same_term[l][j] * ( s6_lxkx_lyky[l][j][0] + 1)

                        d2phidxi_lxky = - a12 * (-12)*(-14) * (s12_lxky_lykx[l][j] * delta_xi[l][j][0] * delta_xi[l][j][1]) \
                                        + a6 * (-6)*(-8) * (s6_lxky_lykx[l][j] * delta_xi[l][j][0] * delta_xi[l][j][1])

                        d2phidxi_lykx = - a12 * (-12)*(-14)*(s12_lxky_lykx[l][j] * delta_xi[l][j][0]* delta_xi[l][j][1]) \
                                        + a6 * (-6)*(-8)* (s6_lxky_lykx[l][j] * delta_xi[l][j][0] * delta_xi[l][j][1])

                        d2phidxi_lyky = - a12 *(-12)* s12_same_term[l][j] * ( s12_lxkx_lyky[l][j][1]  + 1) \
                                        + a6 *(-6)* s6_same_term[l][j]  * ( s6_lxkx_lyky[l][j][1] + 1)

                        d2phidxi_lk = torch.tensor((d2phidxi_lxkx[0],d2phidxi_lxky[0],d2phidxi_lykx[0],d2phidxi_lyky[0])).reshape(2,2)
                        print('l != k d2phidxi_lk',d2phidxi_lk)

                        j = j + 1
                    d2phidxi2_append.append(d2phidxi_lk)
                    #print('d2phidxi2_append',d2phidxi2_append)

                if k == N_particle-1:
                    temp = torch.stack(d2phidxi2_append,dim=0)
                    #print('temp',temp)
                    #print('temp', temp.shape)
                    #print('temp',temp.permute((1,0,2)))
                    temp = temp.permute((1,0,2)).reshape(2, N_particle * DIM )
                    print('reshape',temp)
                    d2phidxi2_append = []

                d2phidxi2_ = torch.cat((d2phidxi2_,temp),dim=0)
                #print('d2phidxi2_',d2phidxi2_)

            d2phidxi2[z] = d2phidxi2_

        return d2phidxi2
