#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from torch.utils.data import Dataset
from ..utils.data_util import data_loader
from ..Integrator.linear_integrator import linear_integrator
import os 
import numpy as np

class Hamiltonian_Dataset(Dataset):
    '''Custom class dataset for hamiltonian dataset'''
    
    def __init__(self, temperature : list, samples : int, mode : str, **kwargs):

        '''
        Parameters
        ----------
        temperature : list
            list of temperature to be used for training
        samples : int
            number of samples per temperature sampled
        mode : str
            only train/validation splitting
        **kwargs : configuration
        '''

        uppath = lambda _path, n : os.sep.join(_path.split(os.sep)[:-n])
        base_dir = uppath(__file__, 2)
        init_path = base_dir + '/init_config/'
        N_particle = kwargs['particle']
        seed = kwargs.get('seed', 9372211)  # first: 937162211 second: 937111

        q_list, p_list = data_loader.loadp_q(init_path,
                                        temperature,
                                        samples,
                                        N_particle) # wrapper for dataloader

        _ratio = 1 # train-validation + valid splitting
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

        N = curr_data.shape[0] ;  kwargs['N'] = N # nsamples
        init_q = curr_data[:,0] ; kwargs['pos'] = init_q
        init_vel = curr_data[:,1] ; kwargs['vel'] = init_vel
        init_p = init_vel * kwargs['m']

        print('== inital data ===')
        print(init_q,init_p)
        print('==================')

        self._setting = kwargs
        print(self._setting)
        q_after, p_after = linear_integrator(**self._setting).integrate(kwargs['hamiltonian'],multicpu=False) # using the integrator class
        q_after, p_after = q_after[-1], p_after[-1] # only take the last from the list

        print('== label data ===')
        print(q_after,p_after)
        print('==================')

        #populate the dataset 
        self._dataset = [] # change the data and label here

        for i in range(N):
            data = (init_q[i],init_p[i])
            label = (q_after[i], p_after[i])

            self._dataset.append([data,label])

        print('dataset loaded')
        
    def __len__(self):
        return len(self._dataset)  # samples x data/label x q/p x num. of particles x DIM
    
    def __getitem__(self, idx):
        return (self._dataset[idx][0], self._dataset[idx][1])

