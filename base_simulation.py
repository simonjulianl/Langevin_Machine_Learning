#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 17:30:43 2020

@author: simon
"""

import numpy as np
from abc import ABC, abstractmethod 
import warnings 
import sympy as sym 
from utils.data_util import data_util

class Integration(ABC) : 

    @abstractmethod
    def __init__(self, *args, **kwargs) -> object:           
        ''' initialize a configuration state 
        Parameters
        ----------            
        **kwargs : 
            - N : int
                total number of particles
                default : 1
            - DIM : int
                Dimension of the particles 
                default : 1 Dimension
            - m : float 
                mass of the particle 
                default : 1
            - kB : float
                boltzmann constant
                default : 1 unit
            - Temperatmport sympy as sym ure : float
                Temperature at which the MC Steps are conducted
                default : 1 unit
            -potential : string
                string representation of the potential U(q) where q is the position
                for eg. (q**2 - 1) ** 2.0 + q 
                
            the potential is assumed to be symmetrical around x, y and z axis
                
        Position and Velocity Matrix : N x DIM matrix 
        
        Returns
        -------
        Abstract Base Class, cannot be instantiated
        '''
        #Configuration settings 
        try : 
            try : 
                potential = kwargs['potential']
                
                def potential_wrapper(q) :
                    ''' helper function to generate potential and force using sympy 
                    parameter is input as q to set the local variable for U(q)
                    '''
                    return eval(potential)
        
                q = sym.symbols('q') # the symbol is changed to x to avoid clashing with local variable q
                dUdq = sym.diff(potential_wrapper(q),q) # get the differentiation expression
                
            except :
                raise ValueError('differentiation fails ')
                
            self._configuration = {
                'N' : kwargs['N'],
                'DIM' : kwargs['DIM'],
                'm' : kwargs['m'],
                'potential' : potential,
                'force' : str(dUdq), # this is using sympy symbol
            }
          
        except : 
            raise ValueError('N / DIM / m / potential unset or error')
        
        
        #Constants
        try : 
            constants = {
                'Temperature' : kwargs['Temperature'],
                'kB' : kwargs['kB']
                }
            
            constants['beta'] = 1 / (constants['Temperature'] * constants['kB'])
    
            self._configuration.update(constants)
            
            del constants 
            
        except : 
            raise ValueError('Constant kB / Temperature Setting Error')
            
 
        pos = np.random.uniform(-1, 1, (self._configuration['N'], self._configuration['DIM']))
        vel = np.random.uniform(-1, 1, (self._configuration['N'], self._configuration['DIM']))
        # create random particle initialization of U[-1,1) for q (pos) and v

        if self._configuration['N'] == 1 : 
            warnings.warn('Initial velocity and pos is not adjusted to COM and external force')

        else :
            MassCentre = np.sum(pos,axis = 0) / self._configuration['N']
            VelCentre = np.sum(vel, axis = 0) / self._configuration['N']
            for i in range(self._configuration['DIM']):
                pos[:,i] = pos[:,i] - MassCentre[i]
                vel[:,i] = vel[:,i] - VelCentre[i]
                
        self._configuration['pos'] = pos
        self._configuration['vel'] = vel
    
    @abstractmethod
    def integrate(self):
        pass
    
    def loadp_q(self, path : str, samples : int) :
        '''
        Base Loader function for load p and q given MCMC initialization

        Parameters
        ----------
        path : str
            absolute path to the init file
        samples : int
            how many sampels per temperature stated in the configuration 

        '''
        
        q_list, p_list = data_util.loadp_q(path, 
                                           [self._configuration['Temperature']],
                                           samples,
                                           self._configuration['DIM'])
            
        # the loaded file must contain N X DIM matrix of p and q
        _new_N, _new_DIM = q_list.shape
        
        if self._configuration['N'] != _new_N : 
            warnings.warn('N is changed to the new value of  {}'.format(_new_N))
            self._configuration['N'] = _new_N
            
        if self._configuration['DIM'] != _new_DIM : 
            warnings.warn('DIM is changed to the new value of  {}'.format(_new_DIM))
            self._configuration['DIM'] = _new_N
        
        self._configuration['vel'] = p_list
        self._configuration['pos'] = q_list # set the new value
        
    def get_configuration(self):
        '''getter function for configuration'''
        return self._configuration
    
    def __repr__(self):
        state = 'current state : \n'
        for key, value in self._configuration.items():
            state += str(key) + ': ' +  str(value) + '\n' 
        return state