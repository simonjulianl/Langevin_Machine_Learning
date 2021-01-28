#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
#from tqdm import trange

class linear_integrator:

    def __init__(self, hamiltonian, integrator_method):

        self._hamiltonian = hamiltonian
        self._integrator_method = integrator_method

    def integrate(self, phase_space, MD_iterations, nsamples, nparticle, DIM, tau_cur, boxsize):

        q_list = torch.zeros((MD_iterations, nsamples, nparticle, DIM))
        p_list = torch.zeros((MD_iterations, nsamples, nparticle, DIM))

        # for i in trange(self._state['MD_iterations']):
        for i in range(MD_iterations):

            q_list[i], p_list[i]  = self._integrator_method(self._hamiltonian, phase_space, tau_cur, boxsize)

        # q_list = q_list[-1]; p_list = p_list[-1]  # only take the last from the list

        if (torch.isnan(q_list).any()) or (torch.isnan(p_list).any()):
            raise ArithmeticError('Numerical Integration error, nan is detected')

        return (q_list, p_list)

