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

    # def save_object(self, qp_list):
    #     with open('test.pt', 'wb') as handle: # overwrites any existing file
    #         pickle.dump(qp_list, handle)

    def step(self, hamiltonian, phase_space, MD_iterations, nsamples_cur, tau_cur):

        nparticle = MC_parameters.nparticle
        DIM =  MC_parameters.DIM
        boxsize =  MC_parameters.boxsize
        iteration_batch = MD_parameters.iteration_batch

        print('step nsamples_cur, tau_cur, MD_iterations')
        print(nsamples_cur, tau_cur, MD_iterations)

        qp_list = []

        for i in range(MD_iterations):

            # print('iteration', i)

            q_list_, p_list_  = self._integrator_method(hamiltonian, phase_space, tau_cur, boxsize)

            qp_stack = torch.stack((q_list_, p_list_))
            # qp = {'q_{}'.format(i): q_list, 'p_{}'.format(i): p_list}
            qp_list.append(qp_stack)

            if i % iteration_batch == iteration_batch - 1:

                with open('test_{}.pt'.format(i // iteration_batch), 'wb') as handle:

                    pickle.dump(qp_list, handle)

                    del qp_list
                    qp_list = []

        for j in range(iteration_batch):

            with open('test_{}.pt'.format(j), 'rb') as handle:  # overwrites any existing file

                a = pickle.load(handle)
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

        # remove files
        for z in range(iteration_batch):
            os.remove('test_{}.pt'.format(z))

        # q_list = q_list[-1]; p_list = p_list[-1]  # only take the last from the list

        if (torch.isnan(q_cat).any()) or (torch.isnan(p_cat).any()):
            index = torch.where(torch.isnan(q_cat))
            print(q_cat[index])
            raise ArithmeticError('Numerical Integration error, nan is detected')

        return (q_cat, p_cat)

