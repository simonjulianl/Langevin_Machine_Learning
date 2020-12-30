#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 20:59:47 2020

@author: simon
"""

import numpy as np
from collections import defaultdict
from itertools import product
import random

class data_loader:
    '''
    Class of helper functions to plot various statistics 
    '''
    @staticmethod # don't need the self to be passed as the first argument
    def loadp_q(path : str, temperature : list , samples : int, particle : int):
        '''
        Function to load p and q based on available files
        N and DIM will be adjusted to loaded p and q 
        Parameters
        ----------
        path : str
            path to initialization folder
            
            default file name : 
                eg. q_N1000_T1_DIM1.npy 
                means the file contains q for 10 000 samples at T = 1, kB is assumed to be 1
                at DIM = 1
                
        temperature : list
            list of temperature to be loaded
            
        samples : int
            total samples per temperature included
            
        DIM : int
            dimension of the particles to be loaded
            
        Raises
        ------
        Exception
            Failed Initialization, unable to find file / load file
        
        Precaution
        ----------
        For Fractional temp, please save the file using - instead of . for decimals
            
        Return 
        ------
        q_list : np.array
            array of loaded q_list of N X DIM matrix
            
        p_list : np.array
            array of loaded p_list of N X DIM matrix 
        '''
        
        import os 
        
        if not os.path.exists(path) :
            raise Exception('path doesnt exist')
            
        _total_particle = 50000
        rho = 0.2
        file_format = path + 'N_particle' + str(particle) +'_samples' + str(_total_particle) + '_rho{}_T{}_pos_sampled.npy'
            # by default each file has 10000 samples
        
        #if samples > _total_particle :
        #    raise Exception('Samples exceeding {} is not allowed'.format(_total_particle))
            
        q_list = None 
        p_list = None
       
        for i, temp in enumerate(temperature) :

            phase_space = np.load(file_format.format(rho,temp))
            curr_q, curr_p = phase_space[0][:samples], phase_space[1][:samples]

            # truncate according to samples
            if i == 0 : # first iteration, copy the data 
                q_list = curr_q
                p_list = curr_p
            else :
                q_list = np.concatenate((q_list, curr_q))
                p_list = np.concatenate((p_list, curr_p))
            
            assert q_list.shape == p_list.shape # shape of p and q must be the same

        return (q_list, p_list)
    
    @staticmethod
    def split_data(q_list, p_list, ratio : float,  seed = 937162211):

        random.seed(seed)  # set the seed
        np.random.seed(seed)

        assert q_list.shape == p_list.shape

        shuffled_indices = np.array(range(q_list.shape[0]),dtype=int)
        np.random.shuffle(shuffled_indices)

        q_list = q_list[shuffled_indices]
        p_list = p_list[shuffled_indices]

        init_pos = np.expand_dims(q_list, axis=1)
        init_vel = np.expand_dims(p_list, axis=1)  # N X 1 X  DIM
        init = np.concatenate((init_pos, init_vel), axis=1)  # N X 2 XDIM

        train_data = init[:int(ratio*init.shape[0])]
        valid_data = init[int(ratio*init.shape[0]):]

        return train_data, valid_data

    @staticmethod
    def test_data(path : str, temperature : str , samples : int, particle : int):

        import os

        if not os.path.exists(path):
            raise Exception('path doesnt exist')

        _total_particle = 5000
        rho = 0.2
        file_format = path + 'N_particle' + str(particle) + '_samples' + str(_total_particle) + '_rho{}_T{}_pos_sampled_test.npy'


        phase_space = np.load(file_format.format(rho, temperature))
        q_list, p_list = phase_space[0][:samples], phase_space[1][:samples]
        q_list = q_list.reshape(-1, q_list.shape[1] * q_list.shape[2])
        p_list = p_list.reshape(-1, p_list.shape[1] * p_list.shape[2])
        init_pos = np.expand_dims(q_list, axis=1)
        init_vel = np.expand_dims(p_list, axis=1)  # N X 1 X  DIM
        data = np.concatenate((init_pos, init_vel), axis=1)  # N X 2 XDIM

        return data

                