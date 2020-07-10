#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 10:21:28 2020

@author: simon
"""

import numpy as np
import copy

class phase_space :
    '''phase space container class that have a 
    q and p configuration as well wrapper to read and write'
    q and p must be either numpy or torch 
    '''
    
    def __init__(self):
        '''initialize phase space container of N X particle X DIM dimension'''
        self._q_list = None
        self._p_list = None

    def set_p(self, p_list):
        print('phase_space.py p_list', p_list[:4])
        self._p_list = copy.deepcopy(p_list)
        print('phase_space.py p_list copy', self._p_list[:4])

    def set_q(self, q_list):
        print('phase_space.py q_list',q_list[:4])
        self._q_list = copy.deepcopy(q_list)
        print('phase_space.py q_list copy', self._q_list[:4])
    
    def get_p(self):
        print('phase_space.py _p_list', self._p_list)
        print('phase_space.py _p_list copy', copy.deepcopy(self._p_list))
        return copy.deepcopy(self._p_list) # N X particle X DIM array


    def get_q(self):
        print('phase_space.py _q_list', self._q_list)
        print('phase_space.py _q_list copy' , copy.deepcopy(self._q_list))
        return copy.deepcopy(self._q_list) # N X particle X DIM array

    def read(self, filename, samples = -1):
        '''function to read the phase space file, 
        the phase space numpy is arranged in q_list ,p_list 
        
        Parameters
        ----------
        filename : str 
            file to be read for phase space
        samples : int
            samples per file , default everything (-1)
        '''
        #print('phase_space.py filename',filename)
        phase_space = np.load(filename)
        #print('phase_space.py phase_space',phase_space.shape)
        #print('phase',phase_space.shape)
        self._q_list = np.array(phase_space[:samples]) # cast to numpy just in case its pickled obj
        #print(self._q_list.shape)
        self._p_list = np.array(self._p_list[:samples])
        #print(self._p_list.shape)
        
        try : 
            assert self._q_list.shape == self._p_list.shape
        except : 
            raise Exception('does not have shape method or shape differs')
         # assert they have the same 
    
    def write(self, filename):
        '''
        function to write the phase space in a numpy file

        Parameters
        ----------
        filename : str
            path to be saved 
        '''
        phase_space = np.array((self._q_list, self._p_list))
        try : 
            assert self._q_list.shape == self._p_list.shape
        except : 
            raise Exception('does not have shape method or shape differs')
            
        np.save(filename, phase_space)
