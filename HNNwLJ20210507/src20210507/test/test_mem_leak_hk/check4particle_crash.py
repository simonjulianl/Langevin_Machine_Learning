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
        crash_id : count when crash
        '''
        self.backward_method = backward_method
        self.ethrsh = ethrsh
        self.pthrsh = pthrsh
        self.crash_id = 0
        print('check4particle_crash initialized : ethrsh ',ethrsh,' pthrsh ',pthrsh)

    # ============================================
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
        energy_per_particle = hamiltonian.total_energy(phase_space) / nparticle

        all_idx = []
        crash_flag = False

        out_of_box = (torch.abs(q_list) > 0.5 * boxsize) # check whether out of boundary

        if out_of_box.any() == True :

            # s_idx is the tuple, each index tensor contains indices for a certain dimension.
            s_idx = torch.where(out_of_box) # condition (BoolTensor)

            # in s_idx, first index tensor represent indices for dim=0 that is along nsamples
            # remove duplicate values that are indices in s_idx
            s_idx = torch.unique(s_idx[0], sorted=True)

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


        check_e = (energy_per_particle > self.ethrsh)

        if check_e.any() == True:

            s_idx = torch.where(check_e) # take sample index; tensor to int => s_idx[0].item()
            s_idx = torch.unique(s_idx[0], sorted=True)
            print('energy_per_particle too high: ',energy_per_particle[s_idx], 'sample idx is ', s_idx)
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
            self.crash_id += 1
            all_idx = torch.cat(all_idx)
            all_idx = torch.unique(all_idx)
            print('all_idx', all_idx)
            #q_list_1, p_list_1 = self.backward_method(hamiltonian,phase_space,tau,boxsize) # HK
            #q_list_2, p_list_2 = self.backward_method(hamiltonian,phase_space,tau,boxsize) # HK
            q_list_back, p_list_back = self.backward_method(hamiltonian,phase_space,tau,boxsize) # HK
            all_q = q_list_back[all_idx] # HK
            all_p = p_list_back[all_idx] # HK
            self.save_crash_config(all_q,all_p,boxsize)
            print('saving qp_state_before_crash for ',all_idx)
            
            full_set = set(range(p_list.shape[0]))
            print('full', len(full_set))
            crsh_set = set(all_idx.tolist()) 
            print('crsh', len(all_idx))
            diff_set = full_set - crsh_set
            print('diff',len(diff_set))

            diff_list = list(diff_set)
            diff_q = q_list[diff_list]
            diff_p = p_list[diff_list]

            phase_space.set_q(diff_q)
            phase_space.set_p(diff_p)
            #print('continuing with these samples ',diff_list)

            return crsh_set 

    # ======================================
    def save_crash_config(self,crash_q_list,crash_p_list,boxsize):
        ''' log crash state in a new file every time crashed '''

        now = datetime.now()
        dt_string = now.strftime("%Y%m%d%H%M%S")
        crash_path = 'crash_dir/'
        crash_filename = crash_path + 'crashed_at' + dt_string + '_' + str(self.crash_id) + '.pt'
        
        # write q, p list # HK
        qp_list = torch.stack((crash_q_list, crash_p_list), dim=1)
        qp_list = torch.unsqueeze(qp_list,dim=2)
        data = {'qp_trajectory': qp_list, 'tau_short': -1, 'tau_long': -1, 'boxsize': boxsize}
        torch.save(data, crash_filename)
        # # shape is [crashed nsamples, (q,p), 1, npaticle, DIM]
        
