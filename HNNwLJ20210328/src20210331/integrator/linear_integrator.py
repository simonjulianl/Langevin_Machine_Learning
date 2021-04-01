import torch
import math
from parameters.MC_parameters import MC_parameters
from utils.crash_log_and_quit import crash_log_and_quit

class linear_integrator:

    ''' linear_integrator class to help implement numerical integrator at each step '''

    _obj_count = 0

    def __init__(self, integrator_method, integrator_method_backward, ethrsh = MC_parameters.max_energy):

        linear_integrator._obj_count += 1
        assert (linear_integrator._obj_count == 1),type(self).__name__ + " has more than one object"

        self._integrator_method = integrator_method
        self._integrator_method_backward = integrator_method_backward
        self.boxsize = MC_parameters.boxsize
        self.ethrsh = ethrsh
        self.pthrsh = math.sqrt( -1. * math.log(math.sqrt(2*math.pi)*1e-6))
        print('pthrsh', self.pthrsh)

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

        energy = hamiltonian.total_energy(phase_space) / MC_parameters.nparticle

        check_q_out = (torch.abs(q_list) > 0.5 * self.boxsize) # check whether out of boundary
        q_nan = torch.isnan(q_list); p_nan = torch.isnan(p_list) # q or p nan error

        if check_q_out.any() == True :

            # s_idx is the tuple, each index tensor contains indices for a certain dimension.
            s_idx = torch.where(check_q_out) # condition (BoolTensor)

            # in s_idx, first index tensor represent indices for dim=0 that is along nsamples
            # remove duplicate values that are indices in s_idx
            s_idx = torch.unique(s_idx[0], sorted=True)
            print('q pbc not applied error in box that is abs boundary', 0.5 * self.boxsize, 'sample idx is ', s_idx)

            # print to file and then quit
            q_list, p_list = self._integrator_method_backward(hamiltonian,phase_space,tau_cur,self.boxsize)
            crash_log_and_quit(q_list[s_idx], p_list[s_idx])

        if (q_nan.any() or p_nan.any()) == True :

            s_idx = (torch.where(q_nan) or torch.where(p_list[p_nan]))
            s_idx = torch.unique(s_idx[0], sorted=True)
            print('q or p nan error', 'sample idx is ', s_idx)

            # print to file and then quit
            q_list, p_list = self._integrator_method_backward(hamiltonian,phase_space,tau_cur,self.boxsize)
            crash_log_and_quit(q_list[s_idx], p_list[s_idx])

        check_p = (p_list > self.pthrsh)
        check_e = (energy > self.ethrsh)

        if check_e.any() == True:

            s_idx = torch.where(check_e) # take sample index; tensor to int => s_idx[0].item()
            s_idx = torch.unique(s_idx[0], sorted=True)
            print('energy too high: ',energy[s_idx], 'sample idx is ', s_idx)

            q_list, p_list = self._integrator_method_backward(hamiltonian,phase_space,tau_cur,self.boxsize)
            crash_log_and_quit(q_list[s_idx], p_list[s_idx])

        if check_p.any() == True:

            s_idx = torch.where(check_p)
            s_idx = torch.unique(s_idx[0], sorted=True)
            print('momentum too high: ', len(p_list[s_idx]), 'sample idx is ', s_idx)

            q_list, p_list = self._integrator_method_backward(hamiltonian,phase_space,tau_cur,self.boxsize)

            crash_log_and_quit(q_list[s_idx], p_list[s_idx])

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

