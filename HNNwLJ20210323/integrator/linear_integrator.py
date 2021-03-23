#!/usr/bin/env python3

import torch
import pickle
import psutil
import gzip
from parameters.MC_parameters import MC_parameters
from parameters.MD_parameters import MD_parameters


class linear_integrator:

    ''' linear_integrator class to help implement numerical integrator at each step '''

    _obj_count = 0

    def __init__(self, integrator_method):

        linear_integrator._obj_count += 1
        assert (linear_integrator._obj_count == 1),type(self).__name__ + " has more than one object"

        self._integrator_method = integrator_method
        self.nparticle = MC_parameters.nparticle
        self.boxsize = MC_parameters.boxsize
        self.iteration_batch = MD_parameters.iteration_batch

    def save_object(self, qp_list, filename, nfile):
        with gzip.open( filename + '_{}.pt'.format(nfile), 'wb') as handle: # overwrites any existing file
            pickle.dump(qp_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()

    def load_object(self, filename, nfile):
        with gzip.open( filename + '_{}.pt'.format(nfile), 'rb') as handle: # overwrites any existing file
            p = pickle.load(handle)

            return p


    def step(self, hamiltonian, phase_space, MD_iterations, nsamples_cur, tau_cur):

        ''' function to save file in case n iterations otherwise return state q, p as prediction

        Parameters
        ----------
        iteration pair batch : int
                iterations for save files
        iteration_batch : int
                multiples of iteration_pair_batch
                1 : train/valid
                int(max_ts) : test
        MD_iterations : int
                n : integrations of short time step
                1 : integration of large time step
        tau_cur : float
                large time step for prediction
                short time step for label
        tau_pair : int
                paired step with two time steps (large and short)

        '''

        # print('integrators tau_cur', tau_cur)
        # print('integrators MD_iterations', MD_iterations)
        iteration_pair_batch = self.iteration_batch * MD_parameters.tau_pair
        # print('iteration_pair_batch', iteration_pair_batch)

        filename = 'tmp/nparticle{}'.format(self.nparticle)

        qp_list = []

        for i in range(MD_iterations):

            # print('iteration', i)

            q_list_, p_list_  = self._integrator_method(hamiltonian, phase_space, tau_cur, self.boxsize)

            qp_stack = torch.stack((q_list_, p_list_))

            bool1_ = phase_space.debug_pbc_bool(q_list_, self.boxsize)
            bool2_ = phase_space.debug_nan_bool(q_list_, p_list_)
            # print('bool', bool1_, bool2_)

            if bool1_.any() == True or bool2_ is not None:
                print('bool true i', i)
                print('pbc not applied or nan error')
                # print(q_list_, p_list_)

            if MD_iterations == 1: # one time step for prediction

                q_list_ = torch.unsqueeze(q_list_, dim=0)
                p_list_ = torch.unsqueeze(p_list_, dim=0)

                return q_list_, p_list_

            if (i+1) % MD_parameters.tau_pair == 0 :
                # add qp_stack paired with large time step that to the list
                qp_list.append(qp_stack)
                # print('i', i, 'memory % used:', psutil.virtual_memory()[2])

            if i % iteration_pair_batch == iteration_pair_batch - 1:

                print('save file', i)
                self.save_object(qp_list, filename, i // iteration_pair_batch )

                del qp_list
                qp_list = []


    def concat_step(self, MD_iterations, tau_cur):

        ''' function to concatenate saved files

        Returns
        ----------
        q list and p list
        '''

        qp_list = []

        iteration_pair_batch = self.iteration_batch * MD_parameters.tau_pair
        filename = 'tmp/nparticle{}'.format(self.nparticle)

        nfile = int(MD_iterations / iteration_pair_batch )

        for j in range(nfile):
            #print('j',j)
            a = self.load_object(filename, j)
            #with open( filename + '_{}.pt'.format(j), 'rb') as handle: # overwrites any existing file
            #   a = pickle.load(handle)

            tensor_a = torch.stack(a)
            print('tensor a', tensor_a.shape)

            q_curr = tensor_a[:,0]
            p_curr = tensor_a[:,1]

            qp_stack = torch.stack((q_curr, p_curr))
            # print(qp_stack[0],qp_stack[1])
            qp_list.append(qp_stack)
            #print('memory % used:', psutil.virtual_memory()[2])

        qp_cat_list = torch.cat(qp_list, dim=1)

        q_list = qp_cat_list[0]
        p_list = qp_cat_list[1]

        if (torch.isnan(q_list).any()) or (torch.isnan(q_list).any()):
            index = torch.where(torch.isnan(q_list))
            print(q_list[index])
            raise ArithmeticError('Numerical Integration error, nan is detected')

        return (q_list, p_list)
