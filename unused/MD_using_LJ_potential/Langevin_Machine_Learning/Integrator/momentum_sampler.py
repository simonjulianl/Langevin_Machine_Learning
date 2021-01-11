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
                number of samples to be sampeld

        Raises
        ------
        Exception
            missing samples in the kwargs

        '''
        super().__init__(**kwargs) 

            
        
    def integrate(self) -> list :

        p_list = self._configuration['phase_space'].get_p()
                
        return p_list 
        
    def __repr__(self):
        state = '\nIntegration Setting : \n'
        for key, value in self._intSetting.items():
            state += str(key) + ': ' +  str(value) + '\n' 
        return state
        
