#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 10:23:21 2020

@author: simon
"""

import numpy as np
import warnings 
from .base_simulation import Integration

class momentum_sampler(Integration):
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
                number of samples to be sampled

        Raises
        ------
        Exception
            missing samples in the kwargs

        '''

        #Seed Setting
        try :
            seed = kwargs['seed']
            np.random.seed(int(seed))
        except :
            warnings.warn('Seed not seed, start using default numpy/random/torch seed')
        # temperature scaling 
            
        
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

        p_list = self._configuration['phase_space'].get_p()
                
        return p_list 
        
    def __repr__(self):
        state = '\nIntegration Setting : \n'
        for key, value in self._intSetting.items():
            state += str(key) + ': ' +  str(value) + '\n' 
        return state
        
