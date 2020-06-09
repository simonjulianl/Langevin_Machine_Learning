#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 13:15:07 2020

@author: simon
"""

import numpy as np
from tqdm import trange
from .base_simulation import Integration
import multiprocessing

class Langevin(Integration):
    '''
    This is langevin implementation with OBABO exact integration based on
    Robust and efficient configurational molecular sampling via Langevin Dynamics
    https://arxiv.org/abs/1304.3269
    '''
    
    def helper(self = None): 
        '''print the common parameters helper'''
        for parent in Langevin.__bases__:
            print(help(parent))
        
    def __init__(self, *args, **kwargs) -> object:
        ''' initialize the NVT Langevin Integration
        
        Parameters
        ----------
        
        **kwargs : integration setting 
        
        iterations : int
            total number of Monte Carlo Steps
            
        DumpFreq : int
            Sampling frequency 
            default : 1 step ( Sample every step )
            
        gamma : float 
            Dampling constant of the langevin differential equation
            
        time_step : float 
            discrete time step of the integration 
            
        seed : float, optional.
            seed for numpy random. default is 937162211
            
        ! user helper function to get full parameter setting !
        
        Returns
        -------
        MSMC Class Object
        
        '''

        super(Langevin, self).__init__(*args, **kwargs)
                
        #Configuration settings 
        try : 
            self._intSetting = {
                'iterations' : kwargs['iterations'],
                'DumpFreq' : kwargs['DumpFreq'],
                'gamma' : kwargs['gamma'],
                'time_step' : kwargs['time_step'],
                'integrator_method' : kwargs['integrator_method'],
                }
            
            if not self._intSetting['iterations'] >= self._intSetting['DumpFreq'] :
                raise ValueError('DumpFreq must be smaller than iterations')

        except : 
            raise TypeError('Integration setting error ( iterations / DumpFreq / gamma / time_step /integrator_method )')
            
        #Seed Setting
        seed = kwargs.get('seed', 937162211)
        np.random.seed(seed)

    def integrate(self):
        '''
        Implementation of OBABO Langevin NVT Sampling 
        
        Precaution
        ----------
        DumpFreq : int 
            Dumping Frequency here acts as the repeater of BAB process
            meaning, if DumpFreq = 50, O (1x) BAB (50x) O(1x)
            this is to ensure that the same random term is used to obtain mean absolute error (MAE)
   
        Raise
        -------
        ArithmeticError
            Integration error, the time step is unstable and numerical number is too big 
            
        Precaution
        ----------
        Leapfrog is special, such that it takes v1/2 instead of v0 as initialization
        hence all the p_list returned is advanced by 1/2 time step 
        
        Returns
        -------
        q_list : np.array
            list of q particles of Samples X N X DIM matrix
        p_list : np.array
            list of p particles of Sampels X N X DIM matrix

        '''
        #obtain all the constants
        N = self._configuration['N'] # total number of particles
        DIM = self._configuration['DIM']
        total_samples = self._intSetting['iterations'] // self._intSetting['DumpFreq']
        gamma = self._intSetting['gamma']
        kB = self._configuration['kB']
        m = self._configuration['m']
        Temp = self._configuration['Temperature']
        time_step = self._intSetting['time_step']
        integrator_method = self._intSetting['integrator_method']
        
        #for langevin sampling, there 2 random terms
        #be careful of the memory size here
        random_1 = np.random.normal(loc = 0.0, scale = 1.0, size = N * total_samples * DIM).reshape(-1,N,DIM)
        random_2 = np.random.normal(loc = 0.0, scale = 1.0, size = N * total_samples * DIM).reshape(-1,N,DIM)
        
        q_list = np.zeros((total_samples, N, DIM))
        p_list = np.zeros((total_samples, N, DIM))
                        
        def integrate_helper(num, return_dict , **state):    
            ''' helper function for multiprocessing 
            
            Precaution
            ----------
            Max N per Process is 1000, be careful with the memory
            
            Parameters
            ----------
            num : int 
                Number of the process passed into the integrate helper
            
            return_dict : dict
                common dictionary between processes
                
            **state : dict
                split state that is passed into the integrate helper
            '''
            
            total_particle = state['N_split']
          
            q_list_temp = np.zeros((total_samples, total_particle, DIM))
            p_list_temp = np.zeros((total_samples, total_particle, DIM)) 
            
            
            state['N'] = total_particle # update total_particle
            
            for i in trange(total_samples): 
                p = state['vel'] * m
                p = np.exp(-gamma * time_step / 2) * p + np.sqrt(kB * Temp / m * ( 1 - np.exp( -gamma * time_step))) * random_1[i][num:num+total_particle]
                state['vel'] = p / m

                for j in range(self._intSetting['DumpFreq']):
                    state = integrator_method(**state)

                p = state['vel'] * m
                p = np.exp(-gamma * time_step / 2) * p + np.sqrt(kB * Temp / m * ( 1 - np.exp( -gamma * time_step))) * random_2[i][num:num+total_particle]
                state['vel'] = p / m

                q_list_temp[i] = state['pos']
                p_list_temp[i] = state['vel'] * m # sample
           
            return_dict[num] = (q_list_temp, p_list_temp) # stored in q,p order
    
        processes = [] # list of processes to be processed 
        manager = multiprocessing.Manager()
        return_dict = manager.dict() # common dictionary 
       
        curr_q = self._configuration['pos']
        curr_p = self._configuration['vel']

        assert curr_q.shape == curr_p.shape
        
        #split using multiprocessing for faster processing
        for i in range(0,len(curr_q),1000):
            split_state = self._configuration
            split_state['pos'] = curr_q[i:i+1000]
            split_state['vel'] = curr_p[i:i+1000]
            split_state['time_step'] = time_step
            split_state['N_split'] = len(split_state['pos'])
            
            p  = multiprocessing.Process(target = integrate_helper, args = (i, return_dict), kwargs = split_state)
            processes.append(p)
       
        for p in processes :
            p.start()
            
        for p in processes :
            p.join() # block the main thread 
            
        #populate the original q list and p list
        for i in return_dict.keys(): #skip every 1000
            q_list[:,i:i+1000] = return_dict[i][0] # q
            p_list[:,i:i+1000] = return_dict[i][1] # p 
            
        self._configuration['pos'] = q_list[-1] # update current state
        self._configuration['vel'] = p_list[-1] # get the latest code
                
        if (np.isnan(q_list).any()) or (np.isnan(p_list).any()):
            raise ArithmeticError('Numerical Integration error, nan is detected')
            
        return (q_list, p_list)
    
    def __repr__(self):
        state = super().__repr__() 
        state += '\nIntegration Setting : \n'
        for key, value in self._intSetting.items():
            state += str(key) + ': ' +  str(value) + '\n' 
        return state
