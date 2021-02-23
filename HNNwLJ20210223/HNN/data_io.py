import torch
import os
from MC_parameters import MC_parameters
from MD_parameters import MD_parameters

class data_io:

    _obj_count = 0

    def __init__(self, init_path):

        data_io._obj_count += 1
        assert(data_io._obj_count == 1), type(self).__name__ + ' has more than one object'

        self.init_path = init_path

    def loadq_p(self, mode):

        if not os.path.exists(self.init_path) :
            raise Exception('path doesnt exist')

        file_format = self.init_path + 'nparticle' + str(MD_parameters.nparticle) + '_new_nsim' + '_rho{}_T{}_pos_' + str(mode) + '_sampled.pt'

        q_list = None
        p_list = None

        for i, temp in enumerate(MD_parameters.temp_list):

            print('temp', temp)
            q_curr, p_curr = torch.load(file_format.format(MC_parameters.rho, temp))

            if i == 0:
                q_list = q_curr
                p_list = p_curr
            else:
                q_list = torch.cat((q_list, q_curr))
                p_list = torch.cat((p_list, p_curr))

            assert q_list.shape == p_list.shape

        return (q_list, p_list)

    def hamiltonian_dataset(self, mode_qp_list):

        q_list, p_list = mode_qp_list
        # shuffle
        g = torch.Generator()
        g.manual_seed(MD_parameters.seed)

        idx = torch.randperm(q_list.shape[0], generator=g)

        q_list_shuffle = q_list[idx]
        p_list_shuffle = p_list[idx]

        try:
            assert q_list_shuffle.shape == p_list_shuffle.shape
        except:
             raise Exception('does not have shape method or shape differs')

        # print('after shuffle')
        # print(q_list_shuffle, p_list_shuffle)

        init_pos = torch.unsqueeze(q_list_shuffle, dim=1) #  nsamples  X 1 X nparticle X  DIM
        init_vel = torch.unsqueeze(p_list_shuffle, dim=1)  #  nsamples  X 1 X nparticle X  DIM
        init = torch.cat((init_pos, init_vel), dim=1)  # nsamples X 2 X nparticle X  DIM

        return init

    def phase_space2label(self, qp_list, linear_integrator, phase_space, noML_hamiltonian):

        nsamples, qnp, nparticles, DIM = qp_list.shape

        curr_q = qp_list[:,0]
        curr_p = qp_list[:,1]
        # print('phase_space2label input',q_list.shape, p_list.shape)
        # print('nsamples',nsamples)

        # print('===== state at short time step 0.01 =====')
        nsamples_cur = nsamples # train or valid
        tau_cur = MD_parameters.tau_short
        MD_iterations = int(MD_parameters.tau_long / tau_cur)
        nsamples_batch = MD_parameters.nsamples_batch

        q_list = torch.zeros((MD_iterations, nsamples_cur, nparticles, DIM))
        p_list = torch.zeros((MD_iterations, nsamples_cur, nparticles, DIM))
        noML_dHdq_fist = torch.zeros((nsamples_cur, nparticles, DIM))

        for z in range(0, len(curr_q), nsamples_batch):

            # print('z',z)
            phase_space.set_q(curr_q[z:z+nsamples_batch])

            noML_dHdq_fist[z:z+nsamples_batch] = noML_hamiltonian.dHdq(phase_space)

            phase_space.set_q(curr_q[z:z+nsamples_batch])
            phase_space.set_p(curr_p[z:z+nsamples_batch])

            q_list[:,z:z+nsamples_batch], p_list[:,z:z+nsamples_batch] = linear_integrator.step( noML_hamiltonian, phase_space, MD_iterations, nsamples_batch, tau_cur)

        return q_list, p_list, noML_dHdq_fist