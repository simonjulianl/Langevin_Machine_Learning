#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from .pb import pb

class phase_space(pb):
    '''phase space container class that have a 
    q and p configuration as well wrapper to read and write'
    q and p must be either numpy or torch 
    '''
    def helper(self = None):
        '''print the common parameters helper'''
        for parent in phase_space.__bases__:
            print(help(parent))

    _obj_count = 0

    def __init__(self):

        super().__init__()

        phase_space._obj_count += 1
        assert(phase_space._obj_count == 1), type(self).__name__ + ' has more than one object'

        '''initialize phase space container of N X particle X DIM dimension'''
        self._q_list = None
        self._p_list = None

    def set_p(self, p_list):
        # self._p_list = copy.deepcopy(p_list)
        self._p_list = p_list.clone()

    def set_q(self, q_list):
        # self._q_list = copy.deepcopy(q_list)
        self._q_list = q_list.clone()

    def get_p(self):
        # return copy.deepcopy(self._p_list) # nsamples N X particle X DIM array
        return self._p_list.clone()

    def get_q(self):
        # return copy.deepcopy(self._q_list) # nsamples N X particle X DIM array
        return self._q_list.clone()

