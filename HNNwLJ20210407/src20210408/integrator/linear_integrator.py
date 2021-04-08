import torch
import math
# from utils.check4particle_crash import check4particle_crash as crsh_chker
from utils.check4particle_crash_dummy import check4particle_crash_dummy as crsh_chker

class linear_integrator:

    ''' linear_integrator class to help implement numerical integrator at each step '''

    _obj_count = 0

    def __init__(self, integrator_method, integrator_method_backward, ethrsh = 1e3):

        linear_integrator._obj_count += 1
        assert (linear_integrator._obj_count == 1),type(self).__name__ + " has more than one object"

        self._integrator_method = integrator_method

        pthrsh = math.sqrt( -1. * math.log(math.sqrt(2*math.pi)*1e-6))
        self.crash_checker = crsh_chker(integrator_method_backward, ethrsh, pthrsh)

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

        return : qp_list
                shape is [nsamples, (q, p), nparticle, DIM]

        '''
        boxsize = phase_space.get_boxsize()
        q_list, p_list  = self._integrator_method(hamiltonian, phase_space, tau_cur, boxsize)
        self.crash_checker.check(phase_space, hamiltonian, tau_cur)
        qp_list = torch.stack((q_list, p_list), dim=1)

        return qp_list

    # =====================================================

    def nsteps(self, hamiltonian, phase_space, tau_cur, nitr, append_strike):

        ''' to integrate more than one step

        nitr : number of strike steps to save file for MD
        append_strike : the period of which qp_list append

        return : list
                append nxt_qp to the qp list
        '''

        assert (nitr % append_strike == 0), 'incompatible strike and nitr'

        qp_list = []
        for t in range(nitr):
            print('====== step ', t)
            nxt_qp = self.one_step( hamiltonian, phase_space, tau_cur)
            # nxt_qp shape is [nsamples, (q, p), nparticle, DIM]

            if (t+1) % append_strike == 0:
                # print('append stike', nxt_qp)
                qp_list.append(nxt_qp)

        return qp_list

