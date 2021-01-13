#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
from tqdm import trange

class linear_integrator:

    def __init__(self, **kwargs):

        self._state = kwargs

    def integrate(self, Hamiltonian):

        # obtain all the constants
        nsamples = self._state['nsamples_cur']
        nparticle = self._state['nparticle']
        DIM = self._state['DIM']
        integrator_method = self._state['integrator_method']

        q_list = torch.zeros((self._state['MD_iterations'], nsamples, nparticle, DIM))
        p_list = torch.zeros((self._state['MD_iterations'], nsamples, nparticle, DIM))

        for i in trange(self._state['MD_iterations']):

            self._state = integrator_method(Hamiltonian, **self._state)

            q_list[i] = self._state['phase_space'].get_q()
            p_list[i] = self._state['phase_space'].get_p()  # sample

        q_list = q_list[-1]; p_list = p_list[-1]  # only take the last from the list

        if (torch.isnan(q_list).any()) or (torch.isnan(p_list).any()):
            raise ArithmeticError('Numerical Integration error, nan is detected')

        return (q_list, p_list)

