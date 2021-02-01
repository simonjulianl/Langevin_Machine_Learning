#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
from HNNwLJ20210128.parameters import MD_paramaters

#from tqdm import trange

class linear_integrator:

    def __init__(self, integrator_method):

        self._integrator_method = integrator_method

    def integrate(self, hamiltonian, phase_space, MD_iterations, nsamples, tau_cur):

        nparticle = MD_paramaters.MD_parameters.nparticle
        DIM =  MD_paramaters.MD_parameters.DIM
        boxsize =  MD_paramaters.MD_parameters.boxsize

        q_list = torch.zeros((MD_iterations, nsamples, nparticle, DIM))
        p_list = torch.zeros((MD_iterations, nsamples, nparticle, DIM))

        # for i in trange(self._state['MD_iterations']):
        for i in range(MD_iterations):
            # print(i)
            q_list[i], p_list[i]  = self._integrator_method( hamiltonian, phase_space, tau_cur, boxsize)

        # q_list = q_list[-1]; p_list = p_list[-1]  # only take the last from the list

        if (torch.isnan(q_list).any()) or (torch.isnan(p_list).any()):
            index = torch.where(torch.isnan(q_list))
            print(q_list[index])
            raise ArithmeticError('Numerical Integration error, nan is detected')

        return (q_list, p_list)

