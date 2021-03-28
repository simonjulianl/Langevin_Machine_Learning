import torch
import os
from utils.data_io import data_io
from MC_parameters import MC_parameters
from MD_parameters import MD_parameters

class dataset:

    ''' dataset class to help split data for train or valid or test and make label for training'''

    _obj_count = 0

    def __init__(self, init_path):

        dataset._obj_count += 1
        assert(dataset._obj_count == 1), type(self).__name__ + ' has more than one object'

        self.data_io_obj = data_io(init_path)


    def _shuffle(self, q_list, p_list):
        # for internal use only

        idx = torch.randperm(q_list.shape[0])

        q_list_shuffle = q_list[idx]
        p_list_shuffle = p_list[idx]

        try:
            assert q_list_shuffle.shape == p_list_shuffle.shape
        except:
             raise Exception('does not have shape method or shape differs')

        return q_list_shuffle, p_list_shuffle


    def qp_dataset(self, mode, crash_filename = None, shuffle = True):

        ''' given mode, split data for train (shuffle=True) or test (shuffle=False)

        Parameters
        ----------
        mode : str
                train or valid or test
        shuffle : bool, optional
                True when train and valid / False when test

        Returns
        ----------
        inital q and p
        '''

        q_list1, p_list1 = self.data_io_obj.read_init_qp(mode)

        if shuffle:
            q_list1, p_list1 = self._shuffle(q_list1, p_list1)

        if crash_filename is not None:

            q_crash, p_crash = self.data_io_obj.read_crash_qp(crash_filename)

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

        return q_list_shuffle, p_list_shuffle


    def phase_space2label(self, curr_q, curr_p, linear_integrator, phase_space, noML_hamiltonian):

        ''' function to make label for train or valid

         Parameters
        ----------
        nsamples : int
                num. of nsamples ( train or valid)
        nsample_batch : int
                num. of batch for nsamples
        pair_iterations : int
                qp end pts iteration paired with large time step
                nstack is no. of paired pts
                ex) large time step = 0.1, short time step = 0.001
                1 : 100 th step at short time step paired with first large time step
                2 : 200 th step at short time step paired with second lare time step
        MD_iteration : int
                n iterations of short time step paired w large time step

        Returns
        ----------
        q and p end pts paired w q and p at large time step
        '''

        nsamples, nparticles, DIM = curr_q.shape

        tau_cur = MD_parameters.tau_short
        pair_iterations = MD_parameters.nstack
        MD_iterations = MD_parameters.tau_pair
        nsamples_batch = MD_parameters.nsamples_batch

        q_list = torch.zeros((pair_iterations, nsamples, nparticles, DIM), dtype=torch.float64)
        p_list = torch.zeros((pair_iterations, nsamples, nparticles, DIM), dtype=torch.float64)

        for z in range(0, len(curr_q), nsamples_batch):

            phase_space.set_q(curr_q[z:z+nsamples_batch])
            phase_space.set_p(curr_p[z:z+nsamples_batch])

            qp_list  = linear_integrator.step( noML_hamiltonian, phase_space, MD_iterations, tau_cur)
            tensor_qp_list = torch.stack(qp_list) # shape is [MD_iterations, (q,p), nsamples, nparticles, DIM]

            q_list[:, z:z + nsamples_batch], p_list[:, z:z + nsamples_batch] = tensor_qp_list[-1,0], tensor_qp_list[-1,1] # take no. of qp paired pts

        return q_list, p_list