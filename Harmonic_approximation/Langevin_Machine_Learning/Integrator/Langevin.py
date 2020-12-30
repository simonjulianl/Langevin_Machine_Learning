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
import copy 

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
            
        tau : float
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
                'gamma' : kwargs['gamma'],
                'tau' : kwargs['tau'],
                'integrator_method' : kwargs['integrator_method']
                }

        except : 
            raise TypeError('Integration setting error ( iterations / gamma / tau /integrator_method )')
            
        #Seed Setting
        seed = kwargs.get('seed', 937162211)
        np.random.seed(seed)

    def integrate(self,multicpu=True):
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
        N = self._configuration['N'] # total number of samples
        particle = self._configuration['particle']  #ADD
        DIM = self._configuration['DIM']
        iterations = self._intSetting['iterations']
        gamma = self._intSetting['gamma']
        kB = self._configuration['kB']
        m = self._configuration['m']
        Temp = self._configuration['Temperature']
        tau = self._intSetting['tau']
        integrator_method = self._intSetting['integrator_method']
        
        #for langevin sampling, there 2 random terms
        #be careful of the memory size here
        random_1 = np.random.normal(loc = 0.0, scale = 1.0, size = N * iterations *particle* DIM).reshape(-1,N,particle, DIM) # ADD *particle
        random_2 = np.random.normal(loc = 0.0, scale = 1.0, size = N * iterations *particle* DIM).reshape(-1,N,particle, DIM) # ADD *particle

        q_list = np.zeros((iterations, N,particle, DIM))
        p_list = np.zeros((iterations, N,particle, DIM))

        if not multicpu:
            print('Not multicpu')
            for i in trange(iterations):

                p = self._configuration['phase_space'].get_p()
                p = np.exp(-gamma * tau / 2) * p + np.sqrt(kB * Temp / m * (1 - np.exp(-gamma * tau))) * \
                    random_1[i]
                self._configuration['phase_space'].set_p(p)

                for j in range(self._intSetting['DumpFreq']):
                    self._configuration = integrator_method(**self._configuration)

                p = self._configuration['phase_space'].get_p()
                p = np.exp(-gamma * tau / 2) * p + np.sqrt(kB * Temp / m * (1 - np.exp(-gamma * tau))) * \
                    random_2[i]
                self._configuration['phase_space'].set_p(p)

                q_list[i] = self._configuration['phase_space'].get_q()
                p_list[i] = self._configuration['phase_space'].get_p()  # sample

        else:
            print('multicpu')
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

                q_list_temp = np.zeros((iterations, total_particle,particle,  DIM))
                p_list_temp = np.zeros((iterations, total_particle,particle,  DIM))

                state['N'] = total_particle # update total_particle

                for i in trange(iterations):

                    p = state['phase_space'].get_p()
                    p = np.exp(-gamma * tau / 2) * p + np.sqrt(  kB * Temp / m * ( 1 - np.exp( -gamma * tau))) * random_1[i][num:num+total_particle]
                    state['phase_space'].set_p(p)

                    state = integrator_method(**state)

                    p = state['phase_space'].get_p()
                    p = np.exp(-gamma * tau / 2) * p + np.sqrt(  kB * Temp / m   * ( 1 - np.exp( -gamma * tau))) * random_2[i][num:num+total_particle]
                    state['phase_space'].set_p(p)

                    q_list_temp[i] = state['phase_space'].get_q()
                    p_list_temp[i] = state['phase_space'].get_p()  # sample

                return_dict[num] = (q_list_temp, p_list_temp) # stored in q,p order

            processes = [] # list of processes to be processed
            # every process creates own 'manager' and 'return_dict'
            manager = multiprocessing.Manager()
            return_dict = manager.dict() # common dictionary
            curr_q = self._configuration['phase_space'].get_q()
            curr_p = self._configuration['phase_space'].get_p()
            assert curr_q.shape == curr_p.shape # N x particle x DIM

            #split using multiprocessing for faster processing
            #step = len(curr_q) //100
            step = len(curr_q)

            for i in range(0, len(curr_q), step):
                split_state = copy.deepcopy(self._configuration) # prevent shallow copying reference of phase space obj
                split_state['phase_space'].set_q(curr_q[i:i + step])
                split_state['phase_space'].set_p(curr_p[i:i+step])
                split_state['tau'] = tau
                split_state['N_split'] = len(split_state['phase_space'].get_q())

                p  = multiprocessing.Process(target = integrate_helper, args = (i, return_dict), kwargs = split_state)

                processes.append(p)

            for p in processes :
                p.start()

            for p in processes :
                p.join() # block the main thread

            #populate the original q list and p list
            for i in return_dict.keys(): #skip every 1000
                q_list[:,i:i+step] = return_dict[i][0] # q #[interations,samples]
                p_list[:,i:i+step] = return_dict[i][1] # p

            self._configuration['phase_space'].set_q(q_list[-1]) # update current state # End of iterations
            self._configuration['phase_space'].set_p(p_list[-1]) # get the latest code  # End of iterations

        if (np.isnan(q_list).any()) or (np.isnan(p_list).any()):
            raise ArithmeticError('Numerical Integration error, nan is detected')
            
        return (q_list, p_list)
    
    def __repr__(self):
        state = super().__repr__() 
        state += '\nIntegration Setting : \n'
        for key, value in self._intSetting.items():
            state += str(key) + ': ' +  str(value) + '\n' 
        return state

