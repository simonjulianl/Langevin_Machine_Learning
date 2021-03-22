import torch
import os
from MC_parameters import MC_parameters
from MD_parameters import MD_parameters

class data_io:

    ''' data_io class to help load, prepare data and label for ML'''

    _obj_count = 0

    def __init__(self, init_path):

        data_io._obj_count += 1
        assert(data_io._obj_count == 1), type(self).__name__ + ' has more than one object'

        self.init_path = init_path

    def loadq_p(self, mode):

        '''
        Parameters
        ----------
        mode : str
                train or valid or crash_filename or test
        temp_list : list
                temperature for dataset

        Returns
        ----------
        concatenated initial q list and p list at temp_list
        '''

        if not os.path.exists(self.init_path) :
            raise Exception('path doesnt exist')

        file_format = self.init_path + 'nparticle' + str(MC_parameters.nparticle) + '_new_nsim' + '_rho{}_T{}_pos_' + str(mode) + '_sampled.pt'

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


    def _shuffle(self, q_list, p_list):
        # for internal use only

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

        return q_list_shuffle, p_list_shuffle

    def qp_dataset(self, filename, crash_filename = None, shuffle = True):

        ''' function to prepare data for train (shuffle=True) or test (shuffle=False)

        Parameters
        ----------
        filename : str
                train or valid or test
        shuffle : bool, optional
                True when train and valid / False when test

        Returns
        ----------
        inital q and p
        '''

        q_list1, p_list1 = self.loadq_p(filename)
        print('before shuffle', q_list1, p_list1)

        if shuffle:
            q_list1, p_list1 = self._shuffle(q_list1, p_list1)

        print('shuffle ? ', q_list1, p_list1)
        if crash_filename is not None:

            q_crash, p_crash = self.loadq_p(crash_filename)

            print('n. of crash data', q_list1.shape, 'n. of original train data', q_list1.shape)

            y = int(MD_parameters.crash_duplicate_ratio * len(q_list1) / len(q_crash)) # duplicate crash data
            z = len(q_list1) - y * len(q_crash)  # reduced train data

            print('crash duplicate', y, 'reduced train data', z)

            indices = torch.randperm(len(q_list1))[:z]

            q_reduced = q_list1[indices]
            p_reduced = p_list1[indices]

            q_duplicate_crash = q_crash.repeat(y,1,1)
            p_duplicate_crash = p_crash.repeat(y,1,1)
            # print(q_duplicate_crash , p_duplicate_crash )

            q_list2 = torch.cat((q_reduced, q_duplicate_crash.cpu()), dim=0)
            p_list2 = torch.cat((p_reduced, p_duplicate_crash.cpu()), dim=0)

            q_list_shuffle, p_list_shuffle = self._shuffle(q_list2, p_list2)

        else:

            q_list_shuffle = q_list1
            p_list_shuffle = p_list1

        init_pos = torch.unsqueeze(q_list_shuffle, dim=1)  # nsamples  X 1 X nparticle X  DIM
        init_vel = torch.unsqueeze(p_list_shuffle, dim=1)  # nsamples  X 1 X nparticle X  DIM
        init = torch.cat((init_pos, init_vel), dim=1)  # nsamples X 2 X nparticle X  DIM

        return init


    def phase_space2label(self, qp_list, linear_integrator, phase_space, noML_hamiltonian):

        ''' function to prepare label data for train or valid or test

         Parameters
        ----------
        nsamples_cur : int
                num. of nsamples ( train or valid or test)
        nsample_batch : int
                num. of batch for nsamples

        Returns
        ----------
        q and p paired w q and p at large time step
        '''

        nsamples, qnp, nparticles, DIM = qp_list.shape

        curr_q = qp_list[:,0]
        curr_p = qp_list[:,1]

        # print('===== state at short time step 0.01 =====')
        nsamples_cur = nsamples # train or valid
        tau_cur = MD_parameters.tau_short
        MD_iterations = int(MD_parameters.tau_long / tau_cur)
        nsamples_batch = MD_parameters.nsamples_batch

        q_list = torch.zeros((MD_iterations, nsamples_cur, nparticles, DIM), dtype=torch.float64)
        p_list = torch.zeros((MD_iterations, nsamples_cur, nparticles, DIM), dtype=torch.float64)

        for z in range(0, len(curr_q), nsamples_batch):

            # print('z',z)
            phase_space.set_q(curr_q[z:z+nsamples_batch])

            phase_space.set_q(curr_q[z:z+nsamples_batch])
            phase_space.set_p(curr_p[z:z+nsamples_batch])

            linear_integrator.step( noML_hamiltonian, phase_space, MD_iterations, nsamples_batch, tau_cur)
            q_list[:, z:z + nsamples_batch], p_list[:, z:z + nsamples_batch] = linear_integrator.concat_step(MD_iterations, tau_cur)

        # print('label', q_list.shape, p_list.shape)
        return q_list, p_list