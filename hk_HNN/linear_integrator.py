#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# applicable for both no ML and ML based integration method

import numpy as np
import torch
from tqdm import trange

import multiprocessing
import copy


class linear_integrator:

    def __init__(self, **kwargs):

        # # Configuration settings
        # try:
        #     self._intSetting = {
        #         'iterations': kwargs['iterations'],
        #         'gamma': kwargs['gamma'],
        #         'tau': kwargs['tau'],
        #         'integrator_method': kwargs['integrator_method']
        #
        #     }
        #
        # except:
        #     raise TypeError('Integration setting error ( iterations / gamma / tau /integrator_method )')

        # Seed Setting
        # seed = kwargs.get('seed', 937162211)
        # np.random.seed(seed)

        self._configuration = kwargs

    def integrate(self, Hamiltonian):

        # obtain all the constants
        N = self._configuration['N']  # total number of samples
        particle = self._configuration['particle']  # ADD
        DIM = self._configuration['DIM']
        # iterations = self._configuration['iterations']
        # self._configuration['iterations'] = self._configuration['tau'] * iterations
        # tau = self._configuration['tau'] * iterations
        # self._configuration['tau'] = tau
        integrator_method = self._configuration['integrator_method']

        q_list = torch.zeros((self._configuration['iterations'], N, particle, DIM))
        p_list = torch.zeros((self._configuration['iterations'], N, particle, DIM))

        print('Not multicpu')
        print(self._configuration)

        for i in trange(self._configuration['iterations']):
            print('iteration {}'.format(i))
            print('inside',Hamiltonian)
            print('integrator',integrator_method)
            self._configuration = integrator_method(Hamiltonian, **self._configuration)
            #print('linear update',self._configuration)
            q_list[i] = self._configuration['phase_space'].get_q()
            p_list[i] = self._configuration['phase_space'].get_p()  # sample


        #print(q_list,p_list)

        if (torch.isnan(q_list).any()) or (torch.isnan(p_list).any()):
            raise ArithmeticError('Numerical Integration error, nan is detected')

        return (q_list, p_list)

    def __repr__(self):
        state = super().__repr__()
        state += '\nIntegration Setting : \n'
        for key, value in self._configuration.items():
            state += str(key) + ': ' + str(value) + '\n'
        return state

