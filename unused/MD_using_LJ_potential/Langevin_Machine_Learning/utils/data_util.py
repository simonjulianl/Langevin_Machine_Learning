#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import random

class data_loader:
    '''
    Class of helper functions to plot various statistics 
    '''
    @staticmethod # don't need the self to be passed as the first argument
    def loadp_q(path : str, temperature : float , samples : int, particle : int):

        '''
        Parameters
        ----------
        path : str
            path to initialization folder
                
        temperature : list
            list of temperature to be loaded
            
        samples : int
            total samples per temperature included
            
        DIM : int
            dimension of the particles to be loaded

        Return 
        ------
        q_list : np.array
            array of loaded q_list of N (nsample) X N_particle X DIM matrix
            
        p_list : np.array
            array of loaded p_list of N (nsample) X N_particle X DIM matrix
        '''
        
        import os 
        
        if not os.path.exists(path) :
            raise Exception('path doesnt exist')

        rho = 0.1
        file_format = path + 'N_particle' + str(particle) +'_samples' + str(samples) + '_rho{}_T{}_pos_sampled.npy'

        phase_space = np.load(file_format.format(rho,temperature))
        q_list, p_list = phase_space[0][:samples], phase_space[1][:samples]

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

        init_pos = np.expand_dims(q_list, axis=1) # N (nsamples) X 1 X N_particle X  DIM
        init_vel = np.expand_dims(p_list, axis=1)  # N (nsamples) X 1 X N_particle X  DIM
        init = np.concatenate((init_pos, init_vel), axis=1)  # N X 2 X N_particle X  DIM

        train_data = init[:int(ratio*init.shape[0])]
        valid_data = init[int(ratio*init.shape[0]):]

        return train_data, valid_data


                