#!/usr/bin/env python3

import torch
from parameters.MC_parameters import MC_parameters
from parameters.MD_parameters import MD_parameters


class linear_integrator:

    ''' linear_integrator class to help implement numerical integrator at each step '''

    _obj_count = 0

    def __init__(self, integrator_method):

        linear_integrator._obj_count += 1
        assert (linear_integrator._obj_count == 1),type(self).__name__ + " has more than one object"

        self._integrator_method = integrator_method
        self.nparticle = MC_parameters.nparticle
        self.boxsize = MC_parameters.boxsize
        self.temp = MD_parameters.temp_list
        self.iteration_batch = MD_parameters.iteration_batch
        self.paird_step = MD_parameters.tau_pair
        self.iteration_save_batch = int(self.paird_step * self.iteration_batch)


    def step(self, hamiltonian, phase_space, MD_iterations, tau_cur):

        ''' function to save file in case n iterations
            otherwise return state q, p as prediction

        Parameters
        ----------
        MD_iterations : int
                n : integrations for save files at short time step
                1 : integration of large time step
        tau_cur : float
                large time step for prediction
                short time step for label

        return : list

        '''

        qp_list = []

        for i in range(MD_iterations):

            q_list, p_list  = self._integrator_method(hamiltonian, phase_space, tau_cur, self.boxsize)

            qp_stack = torch.stack((q_list, p_list))

            bool_ = phase_space.debug_pbc_bool(q_list, self.boxsize)
            nan = phase_space.debug_nan(q_list, p_list)
            # print('bool or nan', bool_, nan)

            if bool_.any() == True or nan is not None:
                print('debug i', i)
                print('pbc not applied or nan error')
                # print(q_list_, p_list_)

            qp_list.append(qp_stack)


        return qp_list

