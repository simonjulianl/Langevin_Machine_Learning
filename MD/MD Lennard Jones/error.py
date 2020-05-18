#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 13:49:37 2020

@author: simon
"""

#%%
'''code to calculate error between different time steps'''
from sklearn.metrics import mean_squared_error

import numpy as np
import os
import glob
import re 
import matplotlib.pyplot as plt
from plot_distribution import plot 
from tqdm import trange

beta = 1

def L2_error(data1,data2):
    return mean_squared_error(data1,data2)

def KL_divergence(p,q): # KL(p||q)
    assert len(p) == len(q) # for KL divergence to work, they must have same numbers of discretized bins
    
    return sum(p[i] * np.log(p[i]/q[i]) for i in range(len(p)))

def JS_divergence(p,q):
    assert len(p) == len(q)
    
    for i in range(len(p)):
        if p[i] == 0.0:
            p[i] = 1e-10 # error correction since it must be continuous
            
    for i in range(len(q)):
        if p[i] == 0.0:
            p[i] = 1e-10 # error correction since it must be continuous
            
    p_data = np.asarray(p)
    q_data = np.asarray(q)
    p_q = 0.5 * (p_data + q_data)
    return 0.5 * (KL_divergence(p_data, p_q) + KL_divergence(q_data,p_q))

def create_prob_histogram(regex,bins = 16,min_interval = -2,max_interval = 2,plots = False): #filename is in form of regex
    directory = os.path.join(os.getcwd(),'MD_Data_7_100') # change the file name according to database required
    if not os.path.exists(directory):
        raise ValueError('path doesnot exist')
        
    regex_fast = re.compile(regex)
   
    files = glob.glob(directory + '/*.npy')
    matches = [string for string in files if re.match(regex_fast,string)]
 
    data1 = np.array([]) # for fast trajectory

    bins = bins 
    ranges = max_interval - min_interval
    # the first bin is to contain all the number < -2 
    # the last bin is to contain all the number  > 2
    multiplier = bins / ranges
    interval = ranges / bins * multiplier #expand so it has interval of 1

    assert interval == 1
    
    ranges = np.arange(min_interval * multiplier-interval,max_interval * multiplier + interval ,interval)
    hist_data = {}
    
    for bins_range in ranges:
        hist_data[bins_range] = 0
        
    unity_constant = 0
    
    for i in trange(len(matches)) : 
        data = np.load(matches[i]).squeeze()
        data = np.clip(data,-2.00001,2) 
        unity_constant += len(data) 

        for datum in data:
            hist_data[datum * multiplier // interval] += 1
            
        del data
        
    proper_hist_data = {}

    for idx, key in enumerate(list(hist_data.keys())):
        proper_hist_data[ranges[idx]/ multiplier] = hist_data[key] / unity_constant

    #plot distribution
    if plots :
        fig = plt.figure()
        plot.plot_histogram(proper_hist_data, potfunc = 'double_well_asymmetrical',label = 'MD')
        # fig.savefig('Exact vs OBABO.png')
        
    del hist_data
    return proper_hist_data



#%%

# =============================================================================
# Real Distribution Ground Truth
# =============================================================================
def pot_func(points):
    return (points ** 2 - 1) ** 2.0 + points

if __name__ == '__main__':
    bins = 16
    
    x = np.linspace(-2,2,100)
    energy = np.exp(-beta * pot_func(x))
    
    dx = np.array([(x2-x1) for x2,x1 in zip(x[1:],x[:-1])])
    dy = np.array([(y2+y1)/2. for y1,y2 in zip(energy[1:],energy[:-1])])
    Z = np.dot(dx,dy.T)
    pdf = energy / Z
    # plt.plot(x,pdf)
    
    ranges = 4 # 2 - (-2)
    mini_interval = 4. / bins
    interval = [(x1,x1 + mini_interval) for x1 in np.arange(-2,2,mini_interval)]
    interval.insert(0,(-100.0,-2.0))
    interval.append((2.0,100.0)) # tail correction
    
    real_distribution = {}
    for lower_bound,upper_bound in interval:
        area_dx = np.linspace(lower_bound,upper_bound,100) # heuristic choice, 100 and 1000 can match to 7 d.p
        area_dy = np.exp(-beta * pot_func(area_dx)) / Z
        # plt.plot(area_dx,area_dy)
        # plt.xlim(-2,2)
        dx_interval = np.array([(x2-x1) for x2,x1 in zip(area_dx[1:],area_dx[:-1])])
        dy_interval = np.array([(y2+y1)/2.0 for y2,y1 in zip(area_dy[1:],area_dy[:-1])])
        
        lb = -2.25 if lower_bound == -100.0 else lower_bound
        real_distribution[lb] = np.dot(dx_interval,dy_interval.T)
            
#%%
    regex_hist = re.compile('.*x005_trajectory[0-9]+_[0-9]+_1e9\.npy') 
    hist = create_prob_histogram(regex_hist,plots = False)
    create_prob_histogram(regex_hist,bins = 100,plots = True) # plot probability graph
    print('JS divergence (real vs fast): ', JS_divergence(list(hist.values()),list(real_distribution.values())))
# Plotting the distribution from all the positions explored

