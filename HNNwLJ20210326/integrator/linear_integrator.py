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
        self.temp = MD_parameters.temp_list
        self.iteration_batch = MD_parameters.iteration_batch
        self.paird_step = MD_parameters.tau_pair
        self.iteration_save_batch = int(self.paird_step * self.iteration_batch)


    def save_object(self, state_list, filename, nfile):

        ''' function to save file in case n iterations
            ex) when prepare label or find gold standard, need more iterations

        Parameters
        ----------
        state_list : list
                list of (q,p) states paired with large time step
        filename : string
        nfile : saved n. of files ( = MD_iterations / iteration save batch)

        save file every iteration save batch ( = paird_step * iteration_batch )

        '''

        with gzip.open( filename + '_{}.pt'.format(nfile), 'wb') as handle: # overwrites any existing file
            pickle.dump(state_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()

    def load_object(self, filename, file):

        ''' function to load saved file

        Parameters
        ----------
        filename : string
        file : each saved file

        returns
        ----------
        load saved file one by one
        '''

        with gzip.open( filename + '_{}.pt'.format(file), 'rb') as handle: # overwrites any existing file
            p = pickle.load(handle)

            return p


    def step(self, hamiltonian, phase_space, MD_iterations, tau_cur):

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
        paird_step : int
                paired step with two time steps (large and short)
                ex) 100 : iterations that are 100 th step at short time step paired with first large time step
                    in case large ts = 0.1 , short ts = 0.001
        iteration_batch : int
                multiples of iteration save batch
                1 : train/valid,  num : test or more iteration for gold standard ex)int(max_ts)
        iteration save batch : int (paird_step * iteration_batch)
                iterations for save files
        nstack : int
                1 : one time step for prediction
                num : multi time steps for prediction

        '''

        nstack = MD_parameters.nstack

        filename = 'tmp/nparticle{}_T{}_tau{}'.format(self.nparticle, self.temp[0],  tau_cur)

        qp_list = []

        for i in range(MD_iterations):

            q_list_, p_list_  = self._integrator_method(hamiltonian, phase_space, tau_cur, self.boxsize)

            qp_stack = torch.stack((q_list_, p_list_))

            bool_ = phase_space.debug_pbc_bool(q_list_, self.boxsize)
            nan = phase_space.debug_nan(q_list_, p_list_)
            # print('bool or nan', bool_, nan)

            if bool_.any() == True or nan is not None:
                print('debug i', i)
                print('pbc not applied or nan error')
                # print(q_list_, p_list_)

            if MD_iterations == nstack:

                q_list_ = torch.unsqueeze(q_list_, dim=0)
                p_list_ = torch.unsqueeze(p_list_, dim=0)

                return q_list_, p_list_

            if (i+1) % self.paird_step == 0 :

                qp_list.append(qp_stack) # add (q,p) state to the list every paired step
                # print('i', i, 'memory % used:', psutil.virtual_memory()[2])

            if i % self.iteration_save_batch == self.iteration_save_batch - 1:

                print('save file', i)
                self.save_object(qp_list, filename, i // self.iteration_save_batch )

                del qp_list
                qp_list = []


    def concat_step(self, MD_iterations, tau_cur):

        ''' function to concatenate saved files

        Parameters
        ----------
        tensor_a : torch.tensor
                shape is [ iteration_batch, (q,p), nsamples, nparticle, DIM ]
        qp_stack : torch.tensor
                concatenates a sequence of tensors along new dim=0
                shape is [ (q,p), iteration_batch, nsamples, nparticle, DIM ]
        qp_list : list
                list of (q,p) states loaded from saved file j in range nfile
        tensor_qp_list :  torch.tensor
                concatenates the given sequence of tensors in given dim=1
                shape is [ (q,p), iterations, nsamples, nparticle, DIM ]

        ex) when integrate 1000 steps at ts = 0.001 for max ts = 1,
            iteration batch = 5, paired step = 100 ( large ts = 0.1, short ts = 0.001)
            save file every 500 iterations and make 2 saved files
            load each saved file and then concatenate them given dim = 1 that is iteration batch

        Returns
        ----------
        q list : shape is [ iterations, nsamples, nparticle, DIM ]
        p list : shape is [ iterations, nsamples, nparticle, DIM ]
        '''

        qp_list = []

        filename = 'tmp/nparticle{}_T{}_tau{}'.format(self.nparticle, self.temp[0], tau_cur)

        nfile = int(MD_iterations / self.iteration_save_batch )

        for j in range(nfile):
            #print('j',j)
            a = self.load_object(filename, j)

            tensor_a = torch.stack(a)
            print('tensor a', tensor_a.shape)

            q_curr = tensor_a[:,0]
            p_curr = tensor_a[:,1]

            qp_stack = torch.stack((q_curr, p_curr))
            print('qp_stack', qp_stack.shape)
            qp_list.append(qp_stack)
            #print('memory % used:', psutil.virtual_memory()[2])

        tensor_qp_list = torch.cat(qp_list, dim=1)
        print('tensor_qp_list', tensor_qp_list.shape)
        q_list = tensor_qp_list[0]
        p_list = tensor_qp_list[1]

        if (torch.isnan(q_list).any()) or (torch.isnan(q_list).any()):
            index = torch.where(torch.isnan(q_list))
            print(q_list[index])
            raise ArithmeticError('Numerical Integration error, nan is detected')

        return (q_list, p_list)
