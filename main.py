#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 18:39:30 2020

@author: simon
"""

from MCMC import MSMC
from momentum_sampler import sample_momentum
from utils.confStats import confStat
from utils.data_util import data_util
import matplotlib.pyplot as plt
from Langevin import Langevin

if __name__ == "__main__":
    import numpy as np
    
    some_array = np.zeros((10000,1))
    np.save('q_N10000_T1_MCMC.npy', some_array)
    np.save('p_N10000_T1_MCMC.npy', some_array)
    
    configuration = {
        'kB' : 1.0, # put as a constant 
        'Temperature' : 1.0,
        'N' : 1,
        'DIM' : 1,
        'm' : 1,
        'potential' : '(q**2.0 - 1) ** 2.0 + q'  # this is asymmetrical double well
        # 'potential' : '0.5* q **2.0', # simple harmonic motion
        }
    
    integration_setting = {
        'iterations' : 1000,
        'DumpFreq' : 1,
        'dq' : 0.001,
        'seed' : 9654645,
        }
    
    configuration.update(integration_setting)
    
    test = MSMC(1,1, **configuration)
    import os 
    
    init_dir = os.getcwd() + '/init/'
    test.loadp_q(init_dir,100)
    # print(test)
    print(test.get_configuration())
    q_list = test.integrate()
    
    print(test)
    
    integration_setting = {
        'samples' : 1000
        }
    
    configuration.update(integration_setting)
    sample_p = sample_momentum(**configuration)
    p_list = sample_p.integrate()
    data_util.plot_stat(q_list, p_list, mode = 'p_dist', **configuration)
    # print(confStat.force(**test._configuration))
    
    # integration_setting = {
    #     'iterations' : 1,
    #     'DumpFreq' : 1,
    #     'time_step' : 0.5,
    #     'gamma' : 1,
    #     }
    
    # configuration.update(integration_setting)
    
    # test2 = Langevin(**configuration)
    # print(test2.integrate()[0])
    
