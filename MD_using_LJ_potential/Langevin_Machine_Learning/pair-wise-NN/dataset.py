#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 10:13:42 2020

@author: simon
"""



from torch.utils.data import Dataset
from ..utils.data_util import data_loader
from ..Integrator.Langevin import Langevin
import os 
import numpy as np 
import warnings 

class Hamiltonian_Dataset(Dataset):
    '''Custom class dataset for hamiltonian dataset'''
    
    def __init__(self, temperature : list, samples : int, mode : str, **kwargs):
        '''
        Hamiltonian custom dataset using the proposed grid based splitting
        return an item with list of (data, label)

        Parameters
        ----------
        temperature : list
            list of temperature to be used for training
        samples : int
            number of samples per temperature sampled 
        mode : str
            only train/validation splitting
        **kwargs : configuration
            this is the state of the training including
            -DIM : int 
                Dimension of the particles
            -seed : float
                for reproducibility
            - **intSetting for Langevin, please check Langevin.py for complete intSetting
        Raises
        ------
        Exception
            missing attributes

        '''
        #print('dataset',kwargs)
        uppath = lambda _path, n : os.sep.join(_path.split(os.sep)[:-n])
        base_dir = uppath(__file__, 2)
        init_path = base_dir + '/init_config/'
        particle = kwargs['particle']
        seed = kwargs.get('seed', 9372211)  # first: 937162211 second: 937111
        q_list, p_list = data_loader.loadp_q(init_path,
                                             temperature, 
                                             samples, 
                                             particle) # wrapper for dataloader
        
        _ratio = 0.8 # train-validation + valid splitting
        train_data, validation_data = data_loader.split_data(q_list, p_list, _ratio,  seed)

        #change the configuration as needed before integrate
        if mode == 'train' : # for training data 
            curr_data = train_data
            print('generating the training data \n')
            del validation_data
        elif mode == 'validation':
            print('generating the validation data \n')
            curr_data = validation_data
            del train_data
        else : 
            raise Exception('Mode not recognized, only train/validation')

        #print('curr_data',curr_data.shape)
        N = curr_data.shape[0] ;  kwargs['N'] = N # samples
        init_q = curr_data[:,0] ; kwargs['pos'] = init_q # data is arranged in q ,p manner
        init_vel = curr_data[:,1] ; kwargs['vel'] = init_vel
        #print('init_q',init_q.shape)
        #print('init_q',init_q)
        #print('init_vel', init_vel)

        #special cases initialization 
        try :
            integrator_method = kwargs['integrator_method']
            time_step = kwargs['time_step']
        except:
            raise Exception('integrator_method / time_step not found')
        # ===================================================================
        if getattr(integrator_method, 'name').startswith('leapfrog') :  #we need to shift v0 to v1/2
            phase_space = kwargs['phase_space']
            Hamiltonian = kwargs['hamiltonian']
            p_list_dummy = np.zeros(phase_space.get_p().shape) # to prevent KE from being integrated
            p = phase_space.get_p()
            phase_space.set_p(p_list_dummy)
            p = p +  time_step/2  * ( -Hamiltonian.get_derivative_q(phase_space) ) #dp/dt
            kwargs['phase_space'].set_p(p)
            warnings.warn('Leapfrog is used, from now on v is 1/2 step ahead of p for same index')
        # ===================================================================
        #print(kwargs)
        q_after, p_after = Langevin(**kwargs).integrate() # using the integrator class
        self._setting = kwargs # save the setting 

        init_p = init_vel * kwargs['m']
        q_after, p_after = q_after[-1], p_after[-1] # only take the last from the list
        #print('after q,p')
        #print(q_after,p_after)

        # make data flatten for MPL
        init_q = init_q.reshape(-1,init_q.shape[1]*init_q.shape[2])
        init_p = init_p.reshape(-1, init_p.shape[1] * init_p.shape[2])

        q_after = q_after.reshape(-1,q_after.shape[1]*q_after.shape[2])
        p_after = p_after.reshape(-1,p_after.shape[1] * p_after.shape[2])
        #print(init_q,init_p)
        #print(q_after,p_after)


        #populate the dataset 
        self._dataset = [] # change the data and label here 
        for i in range(N): # N = samples not num. of particles
            #print(i)
            data = (init_q[i], init_p[i])
            label = (q_after[i], p_after[i])
            #print('dataset.py',data,label)

            # only make 1 big array instead of array of np.array 
            self._dataset.append([data,label])

        #print('dataset',self._dataset)

        print('dataset loaded')
        
    def __len__(self):
        return len(self._dataset)  # samples x data/label x q/p x num. of particles x DIM
    
    def __getitem__(self, idx):
        return (self._dataset[idx][0], self._dataset[idx][1])
    
    def shift_layer(self):
        '''
        This is a function to change the label to the upper stacked NN
        in the first step, we have q0,p0 --> NN --> q1,p1 
        in the second step, we have q0,p0 --> NN --> q1,p2 --> NN --> q2,p2
        where q2 = q0 + dpdt_1 + dpdt_2 and so on

        '''
        print('dataset shape',np.array(self._dataset).shape)
        DIM = len(self._dataset[0][0][0][0]) #sample the first particle for its property
        N_particle = len(self._dataset[0][0][0])
        #print(DIM)
        #print(N_particle)
        #print(len(self._dataset))

        q_temporary = np.zeros((len(self._dataset), N_particle,DIM))
        p_temporary = np.zeros(q_temporary.shape) # ensure they have the same shape
        #print(q_temporary.shape)

        for i,(original_qp, label_qp) in enumerate(self._dataset) :
            curr_q, curr_p = label_qp
            #print(curr_q.shape)

            q_temporary[i] = curr_q ; p_temporary[i] = curr_p

        #print('q1',q_temporary)
        self._setting['pos'] = q_temporary
        self._setting['vel'] = p_temporary * self._setting['m']
        q_after, p_after = Langevin(**self._setting).integrate() # using the integrator class
     
        q_after, p_after = q_after[-1], p_after[-1] # only take the last from the list

        # make data flatten for MPL
        q_after = q_after.reshape(-1, q_after.shape[1] * q_after.shape[2])
        p_after = p_after.reshape(-1, p_after.shape[1] * p_after.shape[2])

        #print('q1_after',q_after)
        #populate the dataset again by changing the label only
        for i in range(len(self._dataset)):
            label = (q_after[i], p_after[i])
            self._dataset[i][1] = label

