#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 10:23:21 2020

@author: simon
"""

import numpy as np
from base_simulation import Integration
import warnings 
from utils.confStats import confStat

class sample_momentum(Integration):
    '''
    Helper class to sample momentum based on direct integration approach
    '''
    def __init__(self, **kwargs):
        '''
        Initialize the NVT momentum sampler for a fix amount of samples

        Parameters
        ----------
        **kwargs : integration Setting
            samples : int
                number of samples to be sampeld

        Raises
        ------
        Exception
            missing samples in the kwargs

        '''
        super().__init__(**kwargs)
        try :
            self._intSetting = {
                'samples' : kwargs['samples']
                }

        except : 
            raise Exception('samples not found')
        
        #Seed Setting
        try :
            seed = kwargs['seed']
            np.random.seed(int(seed))
        except :
            warnings.warn('Seed not seed, start using default numpy/random/torch seed')
        # temperature scaling 
            
        curr_temp = confStat.temp(**self._configuration) # get current temperature
        lmbda = np.sqrt(self._configuration['Temperature'] / curr_temp)
        
        #lambda constant of (Tn / T0) ^0.5 by adjusting KE where Tn is target temp, T0 is initial temp
        
        self._configuration['vel'] = np.multiply(self._configuration['vel'], lmbda)
        
    def integrate(self) -> list :
        '''
        Static method to generate momentum sample that satisfies boltzmann distribution 
        of the state
        
        Parameters
        ----------        
        **configuration : state setting

            kB : float
                Boltzmann constant of the state
                
            Temperature : float
                Temperature of the state
                
            DIM : int
                Dimension of the configuration state 
                
            m : float
                mass of the particles 

        Returns
        -------
        p_list : np.array ( Total Sample X DIM )
            List of momentum sampled 

        '''
        
        try : 
            DIM = self._configuration['DIM']
            beta = 1 / (self._configuration['kB'] * self._configuration['Temperature'])
            m = self._configuration['m']
            
        except : 
            raise Exception('DIM/ temp / mass / kB not found')
            
        total_sample = self._intSetting['samples'] # get total sample
        
        Z = np.sqrt(2 * m * np.pi / beta ) * 2 #total area under the gaussian curve 
        _dp = 0.005 # the width of the dp stripe
        
        scale = np.sqrt(2 * m * np.log(0.0001) / (-beta)) 
        # set the boundary of sampling when prob is proportional to 0.0001
        
        p_list = np.zeros((total_sample, DIM))
        idx = 0 # index to be sampled
        while idx != total_sample:            
            p_sampled = np.random.uniform(-scale, scale, (DIM)) # range of drawing samples [-scale,scale]
            prob_p = 1 # assumming px, py and pz are independent
            
            if DIM == 1 : #we use velocity distribution in 1D / Boltzmann
                prob_p = ((m * beta)/ (2 * np.pi))**0.5 * np.exp(-beta * p_sampled[0] ** 2.0 / (2 * m))  * _dp
            else :
                #we use Maxwell-Boltzmann Distribution ( Distribution of speed )
                speed = np.linalg.norm(p_sampled)
                prob_p = 4 * np.pi * (beta/(2 * np.pi)) ** 1.5 * speed ** 2.0 *  np.exp(-beta * p_sampled[0] ** 2.0 / (2 * m))  * _dp
                
            alpha = np.random.uniform(0,1)
            if alpha <= (prob_p) : #accepted
                p_list[idx] = p_sampled
                idx += 1
                print('{} has been sampled'.format(idx))
                
        return p_list 
        
    def __repr__(self):
        state = super().__repr__() 
        state += '\nIntegration Setting : \n'
        for key, value in self._intSetting.items():
            state += str(key) + ': ' +  str(value) + '\n' 
        return state
        
