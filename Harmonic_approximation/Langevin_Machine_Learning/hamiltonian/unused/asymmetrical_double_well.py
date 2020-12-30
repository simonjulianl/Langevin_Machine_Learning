#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 12:48:21 2020

@author: simon
"""

# HK : put in a new folder call "unused"

from .Interaction import Interaction

class asymmetrical_double_well(Interaction):
    def __init__(self):
        self._name = 'asymmetrical double well of (q**2.0 - 1)**2.0 + q' 
        super().__init__('(q**2.0 - 1)**2.0 + q')
        print('defunct class, exiting')
        quit()

