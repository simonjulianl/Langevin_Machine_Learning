#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 15:52:06 2020

@author: simon
"""

import numpy as np
from tqdm import trange
import copy
import random
from utils.confStats import confStat 
from base_simulation import Integration
import warnings

class MSMC(Integration):
    '''
    This is a Monte Carlo Molecular Simulation
    Only integrate potential function, sampling is for momentum is done exact 
    This class is only used to generate initial configurations
    '''
    
    def helper(self = None): 
        '''print the common parameters helper'''
        for parent in MSMC.__bases__:
            print(help(parent))
        
    def __init__(self, *args, **kwargs) -> object:
        ''' initialize the NVT MSMC module
        Temperature scaled for initialization 
        
        Parameters
        ----------
        
        **kwargs : integration setting 
        
        iterations : int
            total number of Monte Carlo Steps
            

        DumpFreq : int
            Sampling frequency 
            default : 1 step ( Sample every step )
            
        dq : float 
            random walk of position for Monte Carlo
            
        ! user helper function        self.random_1 = random_1
        self.random_2 = random_2.reshape(-1,self.N)to get full parameter setting !
        
        Returns
        -------
        MSMC Class Object
        
        '''

        super(MSMC, self).__init__(*args, **kwargs)
                
        #Configuration settings 
        try : 
            self._intSetting = {
                'iterations' : kwargs['iterations'],
                'DumpFreq' : kwargs['DumpFreq'],
                'dq' : kwargs['dq'],
                }
            
            if not self._intSetting['iterations'] >= self._intSetting['DumpFreq'] :
                raise ValueError('DumpFreq must be smaller than iterations')

        except : 
            raise TypeError('Integration setting error ( iterations / DumpFreq / dq  )')
            
        # seed setting
        try :
            seed = kwargs['seed']
            np.random.seed(int(seed))
            random.seed(int(seed))
        except :
            warnings.warn('Seed not seed, start using default numpy/random/torch seed')
            
        # temperature scaling 
            
        curr_temp = confStat.temp(**self._configuration) # get current temperature
        lmbda = np.sqrt(self._configuration['Temperature'] / curr_temp)
        
        #lambda constant of (Tn / T0) ^0.5 by adjusting KE where Tn is target temp, T0 is initial temp
        
        self._configuration['vel'] = np.multiply(self._configuration['vel'], lmbda)
    
    def mcmove(self) :
        '''
        Helper function for a Monte Carlo Integration 
        Random Walk with the proposed distribution of U[-dq,dq)
        
        Only integrate potential and not kinetic 

        Returns
        -------
        None
            directly change the current configuration setting

        '''
    
        curr_q = self._configuration['pos']
        o = random.randint(0, len(curr_q) - 1) # randomly pick one particle from the state
        
        
        #eno is the old potential energy configuration
        q_list = self._configuration['pos']
        p_list = np.zeros(q_list.shape) # we pass in zero matrix to prevent KE from accidentally integrated
        eno_p = self._configuration['hamiltonian'].get_Hamiltonian(q_list, p_list)
        
        #perform random step with proposed uniform distribution
        qn = np.array(curr_q[o]) + (np.random.uniform(0,1, np.array(curr_q[o]).shape) - 0.5) * self._intSetting['dq']
        qo = np.array(curr_q[o]) # store old position

        self._configuration['pos'][o] = qn # try the new state
            
        #enn is the new potential energy configuration
        q_list = self._configuration['pos']
        enn_p = self._configuration['hamiltonian'].get_Hamiltonian(q_list, p_list)
     
        #accept with probability proportional di e ^ -beta * delta E
        if random.uniform(0,1) >= np.exp(-self._configuration['beta'] * (enn_p - eno_p)):
            self.rejection += 1 # rejected
            self._configuration['pos'][o] = qo #restore to old position 
        
        
    def integrate(self):
        '''
        Implementation of integration for Monte Carlo Integration 
        
        Raises
        ------
        Exception
            Failed MC Move , check self.mcmove() function

        Returns
        -------
        q_list : np.array ( Total Sample X N X DIM )
            array of list of sampled q obtained

        '''
        self.rejection = 0 # reset the rejection counter to check the integration stats 
        total_samples = self._intSetting['iterations'] // self._intSetting['DumpFreq']
        q_list = np.zeros((total_samples, self._configuration['N'],self._configuration['DIM']))
        
        for i in trange(1, self._intSetting['iterations'] + 1, desc = "simulating"):
            self.mcmove()
            if i % self._intSetting['DumpFreq'] == 0 :
                q_list[i-1] = copy.deepcopy(self._configuration['pos'])
                
        
        print('Rejection Rate : {}%'.format(self.rejection * 100/ self._intSetting['iterations']))
        #print out the rejection rate, recommended rejection 40 - 60 % based on Lit
        
        return q_list
    
    def __repr__(self):
        state = super().__repr__() 
        state += '\nIntegration Setting : \n'
        for key, value in self._intSetting.items():
            state += str(key) + ': ' +  str(value) + '\n' 
        return state

