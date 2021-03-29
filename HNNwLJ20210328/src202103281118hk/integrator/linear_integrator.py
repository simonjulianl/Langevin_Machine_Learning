#!/usr/bin/env python3

import torch
from parameters.MC_parameters import MC_parameters
from parameters.MD_parameters import MD_parameters


class linear_integrator:

    ''' linear_integrator class to help implement numerical integrator at each step '''

    _obj_count = 0

    def __init__(self, integrator_method, thrsh=MC_parameters.max_energy):

        linear_integrator._obj_count += 1
        assert (linear_integrator._obj_count == 1),type(self).__name__ + " has more than one object"

        self._integrator_method = integrator_method
        #self.nparticle = MC_parameters.nparticle
        self.boxsize = MC_parameters.boxsize
        self.thrsh = thrsh

    def one_step(self, hamiltonian, phase_space, tau_cur):

        ''' do one step of time integration

        Parameters
        ----------
        hamiltonian : can be ML or noML hamiltonian
        phase_space : contains q_list, p_list as input for integration
        tau_cur : float
                large time step for prediction
                short time step for label

        return : qp_list

        '''
        print(' = = = phase space q = = = = ', phase_space.get_q())
        print(' = = = phase space p = = = = ', phase_space.get_p())
        print(' = = = start = = = = ')
        q_list, p_list  = self._integrator_method(hamiltonian, phase_space, tau_cur, self.boxsize)
        print(' = = = end   = = = = ')

        energy = hamiltonian.total_energy(phase_space)/MC_parameters.nparticle
        print(' = = = phase space q = = = = ', phase_space.get_q())
        print(' = = = phase space p = = = = ', phase_space.get_p())

        bool_ = phase_space.debug_pbc_bool(q_list, self.boxsize)
        nan = phase_space.debug_nan(q_list, p_list)

        if bool_.any() == True or nan is not None:
            # print('debug i', i)
            print('pbc not applied or nan error')
            quit()

        check_e = energy>self.thrsh
        if check_e.any() == True:
            print('energy too high: ',energy)
            quit()

        qp_list = torch.stack((q_list, p_list))

        return qp_list


    def nsteps(self, hamiltonian, phase_space, tau_cur, nitr, append_strike):

        ''' to integrate more than one step

        hamiltonian : can be ML or noML hamiltonian
        phase_space : contains q_list, p_list as input for integration
        tau_cur : float
                large time step for prediction
                short time step for label
        nitr : number of steps for MD
        append_strike : the period of which qp_list is saved

        return : qp_list
        '''

        assert (nitr % append_strike==0), 'incompatible strike and nitr'

        qp_list = []
        for t in range(nitr):
            print('====== step ',t)
            nxt_qp = self.one_step( hamiltonian, phase_space, tau_cur)
            print('nxt qp', nxt_qp)
            # phase_space.set_q(nxt_qp[0])
            # phase_space.set_p(nxt_qp[1])

            if (t+1) % append_strike == 0:
                # print('append stike', nxt_qp)
                qp_list.append(nxt_qp)

        return qp_list

