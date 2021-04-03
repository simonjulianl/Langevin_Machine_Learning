import torch
import math
from parameters.MC_parameters import MC_parameters
from utils.check4particle_crash import check4particle_crash 
#from utils.check4particle_crash_dummy import check4particle_crash_dummy

class linear_integrator:

    ''' linear_integrator class to help implement numerical integrator at each step '''

    _obj_count = 0

    def __init__(self, integrator_method, integrator_method_backward, ethrsh = MC_parameters.max_energy):

        linear_integrator._obj_count += 1
        assert (linear_integrator._obj_count == 1),type(self).__name__ + " has more than one object"

        self._integrator_method = integrator_method
        self.boxsize = MC_parameters.boxsize

        pthrsh = math.sqrt( -1. * math.log(math.sqrt(2*math.pi)*1e-6))
        self.crash_checker = check4particle_crash(integrator_method_backward, ethrsh, pthrsh, self.boxsize)
        #self.crash_checker = check4particle_crash_dummy(integrator_method_backward,ethrsh,pthrsh,self.boxsize)

    def one_step(self, hamiltonian, phase_space, tau_cur):

        ''' do one step of time integration

        Parameters
        ----------
        hamiltonian : can be ML or noML hamiltonian
        phase_space : contains q_list, p_list as input for integration
        tau_cur : float
                large time step for prediction
                short time step for label
        ethrsh      :  float
                give energy threshold = 1e3
        pthrsh      :  float
                give momentum threshold = sqrt(-log(sqrt(2*pi)*1e-6))

        return : qp_list
                shape is [(q,p), nsamples, nparticle, DIM]

        '''

        q_list, p_list  = self._integrator_method(hamiltonian, phase_space, tau_cur, self.boxsize)

        self.crash_checker.check(phase_space, hamiltonian, tau_cur)

        qp_list = torch.stack((q_list, p_list))

        return qp_list


    def nsteps(self, hamiltonian, phase_space, tau_cur, nitr, append_strike):

        ''' to integrate more than one step

        hamiltonian : can be ML or noML hamiltonian
        phase_space : contains q_list, p_list as input for integration
        tau_cur : float
                large time step for prediction
                short time step for label
        nitr : number of strike steps for MD
        append_strike : the period of which qp_list is saved

        return : qp_list
                shape is [append_strike itr, (q,p), nsamples, nparticle, DIM]
        '''
        print(nitr)
        assert (nitr % append_strike == 0), 'incompatible strike and nitr'

        qp_list = []
        for t in range(nitr):
            print('====== step ', t)
            nxt_qp = self.one_step( hamiltonian, phase_space, tau_cur)
            # print('nxt qp', nxt_qp)

            if (t+1) % append_strike == 0:
                # print('append stike', nxt_qp)
                qp_list.append(nxt_qp)

        return qp_list

