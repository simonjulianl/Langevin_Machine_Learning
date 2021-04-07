import torch
from datetime import datetime


class check4particle_crash:
    '''  this class  use for check debug '''

    def __init__(self,backward_method, ethrsh, pthrsh):
        '''
        backward_method: once a configuration is crashed, back integrate to get the
                         configuration before crash
        ethrsh : threshold for total energy - cannot be too high
        pthrsh : threshold for momentum
        boxsize: size of 2D box for LJ
        '''
        self.backward_method = backward_method
        self.ethrsh = ethrsh
        self.pthrsh = pthrsh

    def check(self, phase_space, hamiltonian, tau):
        '''
        check if any samples have
        write crash configurations into crash file and then quit
        returns None and continue running code if there is no crash configuration
        q_list, p_list shape is [nsamples, nparticles, DIM]
        hamiltonian: any hamiltonian object, valid for both noML or ML
        tau: the time step of integration
        '''

        q_list = phase_space.get_q()
        p_list = phase_space.get_p()
        boxsize = phase_space.get_boxsize()

        nparticle = q_list.shape[1]
        energy = hamiltonian.total_energy(phase_space) / nparticle

        all_idx = []
        crash_flag = False

        out_of_box = (torch.abs(q_list) > 0.5 * boxsize) # check whether out of boundary

        if out_of_box.any() == True :

            # s_idx is the tuple, each index tensor contains indices for a certain dimension.
            s_idx = torch.where(out_of_box) # condition (BoolTensor)

            # in s_idx, first index tensor represent indices for dim=0 that is along nsamples
            # remove duplicate values that are indices in s_idx
            s_idx = torch.unique(s_idx[0], sorted=True)
            #print('q pbc not applied error in box that is abs boundary', 0.5 * self.boxsize, 'sample idx is ', s_idx)

            # print to file and then quit
            all_idx.append(s_idx)
            crash_flag = True


        q_nan = torch.isnan(q_list); p_nan = torch.isnan(p_list)

        if (q_nan.any() or p_nan.any()) == True :

            s_idx = (torch.where(q_nan) or torch.where(p_list[p_nan]))
            s_idx = torch.unique(s_idx[0], sorted=True)
            print('q or p nan error', 'sample idx is ', s_idx)

            # print to file and then quit
            all_idx.append(s_idx)
            crash_flag = True


        check_e = (energy > self.ethrsh)

        if check_e.any() == True:

            s_idx = torch.where(check_e) # take sample index; tensor to int => s_idx[0].item()
            s_idx = torch.unique(s_idx[0], sorted=True)
            print('energy too high: ',energy[s_idx], 'sample idx is ', s_idx)
            all_idx.append(s_idx)
            crash_flag = True


        check_p = (p_list > self.pthrsh)

        if check_p.any() == True:

            s_idx = torch.where(check_p)
            s_idx = torch.unique(s_idx[0], sorted=True)
            print('momentum too high: ', len(p_list[s_idx]), 'sample idx is ', s_idx)
            all_idx.append(s_idx)
            crash_flag = True

        if crash_flag == True:
            all_idx = torch.cat(all_idx)
            all_idx = torch.unique(all_idx)
            q_list, p_list = self.backward_method(hamiltonian,phase_space,tau,boxsize)
            all_q = q_list[all_idx]
            all_p = p_list[all_idx]
            self.save_crash_config(all_q,all_p)
            print('saving qp_state_before_crash and then quit')
            quit()


    def save_crash_config(self,crash_q_list,crash_p_list):
        ''' log crash state in a new file every time crashed '''

        now = datetime.now()
        dt_string = now.strftime("%Y%m%d%H%M%S")
        crash_path = '../data/crash_dir/'
        crash_filename = crash_path + 'crashed_at' + dt_string + '.pt'
        
        # write q, p list
        torch.save(torch.stack((crash_q_list, crash_p_list)), crash_filename)
        
