#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import pickle
import os
from parameters.MC_parameters import MC_parameters
from parameters.MD_parameters import MD_parameters

#from tqdm import trange

class linear_integrator:

    _obj_count = 0

    def __init__(self, integrator_method):

        linear_integrator._obj_count += 1
        assert (linear_integrator._obj_count == 1),type(self).__name__ + " has more than one object"

        self._integrator_method = integrator_method

    def save_object(self, qp_list, filename, nfile):
        with open( filename + '_{}.pt'.format(nfile), 'wb') as handle: # overwrites any existing file
            pickle.dump(qp_list, handle)

    def load_object(self, filename, nfile):
        with open( filename + '_{}.pt'.format(nfile), 'rb') as handle: # overwrites any existing file
            return pickle.load(handle)

    def step(self, hamiltonian, phase_space, MD_iterations, nsamples_cur, tau_cur):

        boxsize =  MC_parameters.boxsize

        q_list = None
        p_list = None

        # for i in trange(self._state['MD_iterations']):
        for i in range(MD_iterations):
            # print(i)
            q_curr, p_curr  = self._integrator_method( hamiltonian, phase_space, tau_cur, boxsize)
            q_curr = torch.unsqueeze(q_curr, dim=0)
            p_curr = torch.unsqueeze(p_curr, dim=0)

            if i == 0:
                q_list = q_curr
                p_list = p_curr
            else:
                q_list = torch.cat((q_list, q_curr))
                p_list = torch.cat((p_list, p_curr))

            assert q_list.shape == p_list.shape

        # q_list = q_list[-1]; p_list = p_list[-1]  # only take the last from the list

        if (torch.isnan(q_list).any()) or (torch.isnan(p_list).any()):
            index = torch.where(torch.isnan(q_list))
            print(q_list[index])
            raise ArithmeticError('Numerical Integration error, nan is detected')

        return (q_list, p_list)


    def tiny_step(self, hamiltonian, phase_space, MD_iterations, nsamples_cur, tau_cur, filename):

        nparticle = MC_parameters.nparticle
        boxsize =  MC_parameters.boxsize
        iteration_batch = MD_parameters.iteration_batch

        print('step nsamples_cur, tau_cur, MD_iterations, iteration_batch ')
        print(nsamples_cur, tau_cur, MD_iterations, iteration_batch)

        qp_list = []

        for i in range(MD_iterations):

            print('iteration', i)

            q_list_, p_list_  = self._integrator_method(hamiltonian, phase_space, tau_cur, boxsize)

            qp_stack = torch.stack((q_list_, p_list_))
            # qp = {'q_{}'.format(i): q_list, 'p_{}'.format(i): p_list}
            qp_list.append(qp_stack)

            if i % iteration_batch == iteration_batch - 1:

                self.save_object(qp_list, filename, i // iteration_batch )

                del qp_list
                qp_list = []


    def concat_tiny_step(self, MD_iterations, filename):

        for j in range(int(MD_iterations / MD_parameters.iteration_batch)):

            a = self.load_object(filename, j)

            for k in range(len(a)):
                q_curr = a[k][0]
                p_curr = a[k][1]

                q_curr = torch.unsqueeze(q_curr, dim=0)
                p_curr = torch.unsqueeze(p_curr, dim=0)

                if k == 0:
                    q_list = q_curr
                    p_list = p_curr
                else:
                    q_list = torch.cat((q_list, q_curr))
                    p_list = torch.cat((p_list, p_curr))

            if j == 0:
                q_cat = q_list
                p_cat = p_list
            else:
                q_cat = torch.cat((q_cat, q_list))
                p_cat = torch.cat((p_cat, p_list))

        if (torch.isnan(q_cat).any()) or (torch.isnan(p_cat).any()):
            index = torch.where(torch.isnan(q_cat))
            print(q_cat[index])
            raise ArithmeticError('Numerical Integration error, nan is detected')

        return (q_cat, p_cat)