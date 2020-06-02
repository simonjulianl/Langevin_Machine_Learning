#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 20:59:47 2020

@author: simon
"""

import matplotlib.pyplot as plt
import numpy as np

class data_loader:
    '''
    Class of helper functions to plot various statistics 
    '''
    @staticmethod 
    def loadp_q(path : str, temperature : list , samples : int, DIM : int):
        '''
        Function to load p and q based on available files
        N and DIM will be adjusted to loaded p and q 
        Parameters
        ----------
        path : str
            path to initialization folder
            
            default file name : 
                eg. q_N1000_T1_DIM1_MCMC.npy 
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
            
        _total_particle = 2500
        file_format = path + '{}_N' + str(_total_particle) + '_T{}_DIM{}_MCMC.npy'
            # by default each file has 10000 samples
        
        if samples > _total_particle :
            raise Exception('Samples exceeding {} is not allowed'.format(_total_particle))
            
        q_list = None 
        p_list = None
       
        for temp in temperature : 
            import math # check whether temperature is fractional
            fraction = math.modf(temp)[0] != 0 # boolean
            temp = str(temp).replace('.','-') if fraction else str(int(temp))
            #for fractional temp, use - instead of . when saving
            curr_q = np.load(file_format.format('q',temp,DIM))[:samples] # truncate according to samples
            curr_p = np.load(file_format.format('p',temp,DIM))[:samples]
            if q_list == None or p_list == None : 
                q_list = curr_q
                p_list = curr_p
            else :
                q_list = np.concatenate((q_list, curr_q))
                p_list = np.concatenate((p_list, curr_p))
            
            assert q_list.shape == p_list.shape # shape of p and q must be the same 
    
        return (q_list, p_list)
    
    
                
    @staticmethod        
    def plot_loss(loss : list, mode : str) :
        '''
        helper function to plot loss

        Parameters
        ----------
        loss : list
            np.array of loss 
        mode : str
            change the label for validation, train, test modes

        Raises
        ------
        Exception
            modes not found
        '''
        if mode not in ['validation', 'train', 'test']:
            raise Exception('mode not found, please check the mode')
            
        if mode == 'validation':
            plt.plot(loss, color = 'blue', label='validation loss')
        elif mode =='train' : 
            plt.plot(loss, color = 'blue', label = 'train loss')
        else : 
            plt.plot(loss, color = 'blue', label = 'test loss')
            
        plt.legend(loc = 'best')
        plt.xlabel('epoch')
        plt.show()       
                