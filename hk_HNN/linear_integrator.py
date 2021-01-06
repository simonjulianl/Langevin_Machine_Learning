#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# applicable for both no ML and ML based integration method

import numpy as np
from tqdm import trange
from .base_simulation import Integration
import multiprocessing
import copy


class linear_integrator(Integration):

    def helper(self=None):
        '''print the common parameters helper'''
        for parent in linear_integrator.__bases__:
            print(help(parent))

    def __init__(self, *args, **kwargs) -> object:

        super(linear_integrator, self).__init__(*args, **kwargs)

        # Configuration settings
        try:
            self._intSetting = {
                'iterations': kwargs['iterations'],
                'gamma': kwargs['gamma'],
                'tau': kwargs['tau'],
                'integrator_method': kwargs['integrator_method']

            }

        except:
            raise TypeError('Integration setting error ( iterations / gamma / tau /integrator_method )')

        # Seed Setting
        seed = kwargs.get('seed', 937162211)
        np.random.seed(seed)

    def integrate(self, multicpu=True):

        # obtain all the constants
        N = self._configuration['N']  # total number of samples
        particle = self._configuration['particle']  # ADD
        DIM = self._configuration['DIM']
        iterations = self._intSetting['iterations']
        tau = self._intSetting['tau']
        integrator_method = self._intSetting['integrator_method']

        q_list = np.zeros((iterations, N, particle, DIM))
        p_list = np.zeros((iterations, N, particle, DIM))

        if not multicpu:
            print('Not multicpu')

            self._configuration.update(self._intSetting)

            for i in trange(iterations):
                self._configuration = integrator_method(**self._configuration)

                q_list[i] = self._configuration['phase_space'].get_q()
                p_list[i] = self._configuration['phase_space'].get_p()  # sample

        else:
            print('multicpu')

            def integrate_helper(num, return_dict, **state):
                ''' helper function for multiprocessing

                Precaution
                ----------
                Max N per Process is 1000, be careful with the memory

                Parameters
                ----------
                num : int
                    Number of the process passed into the integrate helper

                return_dict : dict
                    common dictionary between processes

                **state : dict
                    split state that is passed into the integrate helper
                '''

                total_particle = state['N_split']

                q_list_temp = np.zeros((iterations, total_particle, particle, DIM))
                p_list_temp = np.zeros((iterations, total_particle, particle, DIM))

                state['N'] = total_particle  # update total_particle

                for i in trange(iterations):
                    state = integrator_method(**state)

                    q_list_temp[i] = state['phase_space'].get_q()
                    p_list_temp[i] = state['phase_space'].get_p()  # sample

                return_dict[num] = (q_list_temp, p_list_temp)  # stored in q,p order

            processes = []  # list of processes to be processed
            # every process creates own 'manager' and 'return_dict'
            manager = multiprocessing.Manager()
            return_dict = manager.dict()  # common dictionary
            curr_q = self._configuration['phase_space'].get_q()
            curr_p = self._configuration['phase_space'].get_p()
            assert curr_q.shape == curr_p.shape  # N x particle x DIM

            # split using multiprocessing for faster processing
            # step = len(curr_q) //100
            step = len(curr_q)

            for i in range(0, len(curr_q), step):
                split_state = copy.deepcopy(self._configuration)  # prevent shallow copying reference of phase space obj
                split_state['phase_space'].set_q(curr_q[i:i + step])
                split_state['phase_space'].set_p(curr_p[i:i + step])
                split_state['tau'] = tau
                split_state['N_split'] = len(split_state['phase_space'].get_q())

                p = multiprocessing.Process(target=integrate_helper, args=(i, return_dict), kwargs=split_state)

                processes.append(p)

            for p in processes:
                p.start()

            for p in processes:
                p.join()  # block the main thread

            # populate the original q list and p list
            for i in return_dict.keys():  # skip every 1000
                q_list[:, i:i + step] = return_dict[i][0]  # q #[interations,samples]
                p_list[:, i:i + step] = return_dict[i][1]  # p

            # print(q_list,p_list)
            self._configuration['phase_space'].set_q(q_list[-1])  # update current state # End of iterations
            self._configuration['phase_space'].set_p(p_list[-1])  # get the latest code  # End of iterations

        if (np.isnan(q_list).any()) or (np.isnan(p_list).any()):
            raise ArithmeticError('Numerical Integration error, nan is detected')

        return (q_list, p_list)

    def __repr__(self):
        state = super().__repr__()
        state += '\nIntegration Setting : \n'
        for key, value in self._intSetting.items():
            state += str(key) + ': ' + str(value) + '\n'
        return state

