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
import torch 

class Hamiltonian_Dataset(Dataset):
    '''Custom class dataset for hamiltonian dataset'''
    
    def __init__(self, temperature : list, samples : int, mode : str, **kwargs):
        '''
        Hamiltonian custom dataset using the proposed grid based splitting
        return an item with tensor of (data, label)

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
        uppath = lambda _path, n : os.sep.join(_path.split(os.sep)[:-n])
        base_dir = uppath(__file__, 2)
        init_path = base_dir + '/init/'
        DIM = kwargs['DIM']
        seed = kwargs.get('seed',  937162211)
        q_list, v_list = data_loader.loadp_q(init_path,
                                             temperature, 
                                             samples, 
                                             DIM) # wrapper for dataloader 
        
        _ratio = 0.6 # this splitting ratio is kept constant             
        train_data, validation_data = data_loader.grid_split_data(q_list, v_list, _ratio,  seed)
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
            
        N = curr_data.shape[0] ;  kwargs['N'] = N
        init_q = curr_data[:,0] ; kwargs['pos'] = init_q # data is arranged in q ,p manner
        init_vel = curr_data[:,1] ; kwargs['vel'] = init_vel

        #special cases initialization 
        try :
            integrator_method = kwargs['integrator_method']
            time_step = kwargs['time_step']
        except:
            raise Exception('integrator_method / time_step not found')
        # ===================================================================
        if getattr(integrator_method, 'name').startswith('leapfrog') :  #we need to shift v0 to v1/2
            vel = kwargs['vel'] 
            Hamiltonian = kwargs['hamiltonian']
            p_list_dummy = np.zeros(kwargs['pos'].shape) # to prevent KE from being integrated
            vel = vel +  time_step/2  * ( -Hamiltonian.get_derivative_q(kwargs['pos'], p_list_dummy) ) #dp/dt
            kwargs['vel'] = vel
            warnings.warn('Leapfrog is used, from now on v is 1/2 step ahead of p for same index')
        # ===================================================================
        
        q_after, p_after = Langevin(**kwargs).integrate() # using the integrator class
        
        self._setting = kwargs # save the setting 
        
        init_p = init_vel * kwargs['m']
        q_after, p_after = q_after[-1], p_after[-1] # only take the last from the list
        
        #populate the dataset 
        self._dataset = [] # change the data and label here 
        for i in range(N):
            data = (list(init_q[i]), list(init_p[i]))
            label = (list(q_after[i]), list(p_after[i]))
            # only make 1 big array instead of array of np.array 
            self._dataset.append(torch.tensor([data,label])) # every data is made of torch.tensor
            
        print('dataset loaded')
        
    def __len__(self):
        return len(self._dataset)
    
    def __getitem__(self, idx):
        return (self._dataset[idx][0], self._dataset[idx][1])
    
    def shift_layer(self):
        '''
        This is a function to change the label to the upper stacked NN
        in the first step, we have q0,p0 --> NN --> q1,p1 
        in the second step, we have q0,p0 --> NN --> q1,p2 --> NN --> q2,p2
        where q2 = q0 + dpdt_1 + dpdt_2 and so on

        '''
        DIM = self._dataset[0][0][0].shape[0] #sample the first particle for its property
        q_temporary = np.zeros((len(self._dataset), DIM)) 
        p_temporary = np.zeros(q_temporary.shape) # ensure they have the same shape
        for i,(original_qp, label_qp) in enumerate(self._dataset) :
            curr_q, curr_p = label_qp
            q_temporary[i] = curr_q ; p_temporary[i] = curr_p
         
        self._setting['pos'] = q_temporary ; 
        self._setting['vel'] = p_temporary / self._setting['m']
        q_after, p_after = Langevin(**self._setting).integrate() # using the integrator class
     
        q_after, p_after = q_after[-1], p_after[-1] # only take the last from the list
        #populate the dataset again by changing the label only
        for i in range(len(self._dataset)):
            label = (q_after[i], p_after[i])
            self._dataset[i][1] = label
            
