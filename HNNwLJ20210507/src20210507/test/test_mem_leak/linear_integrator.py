import torch
import math
import psutil
from diagnostics import collect_garbages

class linear_integrator:

    ''' linear_integrator class to help implement numerical integrator at each step '''

    _obj_count = 0

    def __init__(self, integrator_method, crash_chker):

        '''
        parameters
        ___________
        backward_method: once a configuration is crashed, back integrate to get the
                         configuration before crash
        ethrsh: threshold for total energy - cannot be too high
        '''

        linear_integrator._obj_count += 1
        assert (linear_integrator._obj_count == 1),type(self).__name__ + " has more than one object"

        self._integrator_method = integrator_method

        self.crash_checker = crash_chker
        print('linear_integrator initialized ')

    # =================================================

    def one_step(self, hamiltonian, phase_space, tau_cur):

        ''' do one step of time integration

        Parameters
        ----------
        hamiltonian : can be ML or noML hamiltonian
        phase_space : contains q_list, p_list as input for integration and contains boxsize also
        tau_cur : float
                large time step for prediction
                short time step for label
        crash_idx : indices of crashed qp_list
                if crash, return crash indices. otherwise None

        return :torch.tensor
                shape is [nsamples, (q, p), nparticle, DIM]
        '''

        boxsize = phase_space.get_boxsize()
        q_list, p_list  = self._integrator_method(hamiltonian, phase_space, tau_cur, boxsize)
        # q_list shape is [nsamples, nparticle, DIM]
        # print('q, p shape before crash checker ', q_list.shape, p_list.shape)

        crash_idx = self.crash_checker.check(phase_space, hamiltonian, tau_cur) 

        print('q, p shape after crash checker ',phase_space.get_q().shape, phase_space.get_p().shape, 'crash_idx ', crash_idx)

        qp_list = torch.stack((phase_space.get_q(), phase_space.get_p()), dim=1)
        # shape is [nsamples, (q,p), nparticle, DIM]

        return qp_list, crash_idx

    # =====================================================

    def nsteps(self, hamiltonian, phase_space, tau_cur, nitr, append_strike):

        ''' to integrate more than one step

        nitr : number of strike steps to save file for MD
        append_strike : the period of which qp_list append

        return :
        qp_list : list
                append nxt_qp to the qp list
        crash_flag : True or False
                if crash_idx exist, return crash_flag is True 
        '''

        assert (nitr % append_strike == 0), 'incompatible strike and nitr'

        crash_flag = False

        qp_list = []

        for t in range(nitr):
            print('====== step ', t, flush=True)
            nxt_qp, crash_idx = self.one_step( hamiltonian, phase_space, tau_cur)
            # nxt_qp shape is [nsamples, (q, p), nparticle, DIM]

            print('t', t, 'memory % used:', psutil.virtual_memory()[2], '\n')

            if (t+1) % append_strike == 0:

                # if crash_idx is not None, make empty qp_list and crash_flag is True 
                if crash_idx is not None:
                    qp_list = []
                    crash_flag = True

                qp_list.append(nxt_qp)
                #collect_garbages()  # HK


        return qp_list, crash_flag

