#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 12:56:07 2020

@author: simon
"""

    
from .Interaction import Interaction

class SHO_potential(Interaction):
    def __init__(self, k : float):
        '''
        Parameters
        ----------
        k : float
            the spring constant k 
        '''
        try : 
            str_k = str(float(k))
        except : 
            raise Exception('k is not a float / error in spring constant')
        self._name = 'Simple Harmonic Oscillation Potential' 
        super().__init__('1/2 * q ** 2.0 * {}'.format(str_k))