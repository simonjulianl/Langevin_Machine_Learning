#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 17:44:13 2020

@author: simon
"""

import pickle 
dict = {'Python' : '.py', 'C++' : '.cpp', 'Java' : '.java'}
f = open("file.pkl","wb")
pickle.dump(dict,f)
f.close()

f = open("file.pkl","rb")
data = pickle.load(f)
print(data)
# class Hamiltonian:
#     return 

