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

        '''
        Parameters
        ----------
        iteration_batch : int
                multiples of iteration_pair_batch
        paird_steps : int
                paired step with two time steps (large and short)
        iteration save batch : int
                iterations for save files
        '''

        linear_integrator._obj_count += 1
        assert (linear_integrator._obj_count == 1),type(self).__name__ + " has more than one object"

        self._integrator_method = integrator_method
        self.nparticle = MC_parameters.nparticle
        self.boxsize = MC_parameters.boxsize
        self.temp = MD_parameters.temp_list
        self.iteration_batch = MD_parameters.iteration_batch
        self.paird_step = MD_parameters.tau_pair
        self.iteration_save_batch = int(self.paird_step * self.iteration_batch)

    def save_object(self, qp_list, filename, nfile):
        with gzip.open( filename + '_{}.pt'.format(nfile), 'wb') as handle: # overwrites any existing file
            pickle.dump(qp_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()

    def load_object(self, filename, nfile):
        with gzip.open( filename + '_{}.pt'.format(nfile), 'rb') as handle: # overwrites any existing file
            p = pickle.load(handle)

            return p


    def step(self, hamiltonian, phase_space, MD_iterations, nsamples_cur, tau_cur):

        ''' function to save file in case n iterations
            otherwise return state q, p as prediction

        Parameters
        ----------
        MD_iterations : int
                n : integrations of short time step
                1 : integration of large time step
        tau_cur : float
                large time step for prediction
                short time step for label
        nstack : int
                1 : 100 th step at short time step paired with first large time step in case large ts = 0.1 , short ts = 0.001
        '''

        nstack = MD_parameters.nstack

        # print('iteration_pair_batch', iteration_pair_batch)

        filename = 'tmp/nparticle{}_T{}_tau{}'.format(self.nparticle, self.temp[0],  tau_cur)

        # print('step nsamples_cur, tau_cur, MD_iterations, iteration_batch ')
        # print(nsamples_cur, tau_cur, MD_iterations, self.iteration_batch)

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

            if MD_iterations == nstack: # one time step for prediction

                q_list_ = torch.unsqueeze(q_list_, dim=0)
                p_list_ = torch.unsqueeze(p_list_, dim=0)

                return q_list_, p_list_

            if (i+1) % self.paird_step == 0 :
                # add qp_stack paired with large time step that to the list
                qp_list.append(qp_stack)
                # print('i', i, 'memory % used:', psutil.virtual_memory()[2])

            if i % self.iteration_save_batch == self.iteration_save_batch - 1:

                print('save file', i)
                self.save_object(qp_list, filename, i // self.iteration_save_batch )

                del qp_list
                qp_list = []


    def concat_step(self, MD_iterations, tau_cur):

        ''' function to concatenate saved files

        Returns
        ----------
        q list and p list
        '''

        qp_list = []

        filename = 'tmp/nparticle{}_T{}_tau{}'.format(self.nparticle, self.temp[0], tau_cur)

        nfile = int(MD_iterations / self.iteration_save_batch )

        for j in range(nfile):
            #print('j',j)
            a = self.load_object(filename, j)
            #with open( filename + '_{}.pt'.format(j), 'rb') as handle: # overwrites any existing file
            #   a = pickle.load(handle)

            #handle.close()
            tensor_a = torch.stack(a)
            print('tensor a', tensor_a.shape)

            q_curr = tensor_a[:,0]
            p_curr = tensor_a[:,1]

            qp_stack = torch.stack((q_curr, p_curr))
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
