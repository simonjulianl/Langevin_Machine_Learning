#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch

class phase_space :
    '''phase space container class that have a 
    q and p configuration as well wrapper to read and write'
    q and p must be either numpy or torch 
    '''
    
    def __init__(self):
        '''initialize phase space container of N X particle X DIM dimension'''
        self._q_list = None
        self._p_list = None
        self._grid_list = None

    def set_p(self, p_list):
        # self._p_list = copy.deepcopy(p_list)
        self._p_list = p_list.clone()

    def set_q(self, q_list):
        # self._q_list = copy.deepcopy(q_list)
        self._q_list = q_list.clone()

    def set_grid(self, grid_list):
        # self._q_list = copy.deepcopy(q_list)
        self._grid_list = grid_list.clone()
    
    def get_p(self):
        # return copy.deepcopy(self._p_list) # nsamples N X particle X DIM array
        return self._p_list.clone()

    def get_q(self):
        # return copy.deepcopy(self._q_list) # nsamples N X particle X DIM array
        return self._q_list.clone()

    def get_grid(self):
        # return copy.deepcopy(self._q_list) # nsamples N X particle X DIM array
        return self._grid_list.clone()

    def read(self, filename, nsamples):
        '''function to read the phase space file, 
        the phase space numpy is arranged in q_list ,p_list 
        
        Parameters
        ----------
        filename : str 
            file to be read for phase space
        nsamples : int
            nsamples per file , default everything (-1)
        '''
        # phase_space = torch.from_numpy(np.load(filename))
        phase_space = torch.load(filename)
        # print(phase_space.shape)
        self._q_list =  phase_space[0][:nsamples]
        self._p_list =  phase_space[1][:nsamples]

        try : 
            assert self._q_list.shape == self._p_list.shape
        except : 
            raise Exception('does not have shape method or shape differs')
         # assert they have the same 

        return self._q_list, self._p_list

    def build_gridpoint(self, npixels, boxsize, DIM):

        xvalues = torch.arange(0, npixels, dtype=torch.float64)
        xvalues = xvalues - 0.5 * npixels
        yvalues = torch.arange(0, npixels, dtype=torch.float64)
        yvalues = yvalues - 0.5 * npixels
        gridx, gridy = torch.meshgrid(xvalues * (boxsize/npixels) , yvalues * (boxsize/npixels) )
        self._grid_list = torch.stack([gridx, gridy], dim=-1)
        self._grid_list = self._grid_list.reshape((-1, DIM))

        print(self._grid_list)

        return self._grid_list
