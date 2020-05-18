#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 14:02:18 2020

@author: simon
"""

import numpy as np
import matplotlib.pyplot as plt

def potential(points):
    return (points ** 2 -1) ** 2 + points

beta = 1

x = np.linspace(-2,2,100)
y = np.exp(-beta * potential(x))

#normalization constant, we are going to estimate the area under the curve

dx = np.array([(x2-x1) for x2,x1 in zip(x[1:],x[:-1])])
#trapezium
ys=  np.array([(y2+y1) / 2. for y2,y1 in zip(y[1:],y[:-1])])

Z = np.dot(dx,ys.T)
plt.plot(x,y/Z,marker= None, linestyle = '-', label = 'pdf')

plt.grid(True)
plt.title('probability density function')
plt.xlim(-2.5,2.5)
plt.legend(loc = 'best')

plt.figure()
plt.grid(True)
plt.title('Probability')
plt.plot(x,y/sum(y),marker = None, linestyle = '-',label = 'Probability')
plt.legend(loc = 'best')