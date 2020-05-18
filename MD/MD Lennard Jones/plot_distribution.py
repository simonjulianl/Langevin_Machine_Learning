#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 11:04:50 2020

@author: simon
"""
import numpy  as np
import matplotlib.pyplot as plt

class plot:
    @staticmethod
    def potential(points,potfunc):
            if potfunc == 'double_well_symmetrical':
                return ((points-1 ) ** 2 * (points + 1) ** 2)
            elif potfunc == 'double_well_asymmetrical':
                return (points ** 2 -1) ** 2 + points
        
    @staticmethod
    #this plots the probability against an axis
    def distribution(data,bins,min_interval = -2,max_interval = 2,label = 'MD',potfunc = 'double_well_asymmetrical'):
        nbins = bins
        ranges = max_interval - min_interval
        range_marking = np.arange(min_interval,max_interval,ranges/nbins)  #since the interval is [-2,2]
        final_pos = {} 
        mean_correction = (ranges / 2) / nbins
        for bins in range_marking:
            final_pos[bins] = 0
        
        data = data.squeeze()
        for i in range(len(data)):
            histbin = None
            if not (min_interval <= data[i] <= max_interval):
                continue
            
            for j in range(nbins-1,-1,-1):
                if data[i] >= range_marking[j] : 
                    histbin = range_marking[j] 
                    break
                
            final_pos[histbin] += 1
           
        
        beta = 1
        
        x = np.linspace(-2,2,100)
        y = np.exp(-beta * plot.potential(x,potfunc))
        
        #normalization constant, we are going to estimate the area under the curve
        
        dx = np.array([(x2-x1) for x2,x1 in zip(x[1:],x[:-1])])
        #trapezium
        ys=  np.array([(y2+y1) / 2. for y2,y1 in zip(y[1:],y[:-1])])
        
        Z = np.dot(dx,ys.T)
        plt.grid(True)
        plt.xlim(-2.0,2.0)
        plt.title('Probability')
        plt.plot(x,y/sum(y),marker = None, linestyle = '-',label = 'exact')
        plt.plot(np.array(list(final_pos.keys())) + mean_correction,np.array(list(final_pos.values())) / len(data),label = label,linestyle = '-',marker = None,color = 'black' )
        plt.legend(loc = 'best')
        plt.show()
       
        
        return (np.array(list(final_pos.values())) / len(data),np.array(list(final_pos.keys())) + mean_correction)
    
    @staticmethod
    #this plot the distribution given the histogram
    def plot_histogram(data,potfunc = 'double_well_asymmetrical',label = 'MD'):
        #data histogram is in form of dictionary
        assert str(data.__class__).find('dict') != -1
        
        beta = 1
        x = np.linspace(-2,2,100)
        y = np.exp(-beta * plot.potential(x,potfunc))
        
        unity_constant = sum(data.values())
        proper_hist = {}
        if float(unity_constant) != 1 : #is not normalized yet
            for key in data.keys():
                proper_hist[key] = data[key] / unity_constant
                
        plt.grid(True)
        plt.xlim(-2.0,2.0)
        plt.title('Probability')
        plt.plot(x,y/sum(y), marker = None, linestyle = '-', label='exact')
        plt.plot(np.array(list(proper_hist.keys())),np.array(list(proper_hist.values())), label = label, linestyle = '-',marker = None,color = 'black')
        plt.legend(loc = 'best')
        plt.show()
        
        