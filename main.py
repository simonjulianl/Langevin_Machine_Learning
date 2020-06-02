#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 18:39:30 2020

@author: simon
"""

from MCMC import MSMC
from momentum_sampler import sample_momentum
from utils.confStats import confStat
from utils.data_util import data_loader
import matplotlib.pyplot as plt
from Langevin import Langevin
import numpy as np
from hamiltonian.hamiltonian import Hamiltonian
from hamiltonian.SHO_interactions import SHO_interactions
from hamiltonian.SHO_potential import SHO_potential
from hamiltonian.Lennard_Jones import Lennard_Jones

if __name__ == "__main__":

    test = Hamiltonian()
    test.append(SHO_potential(5))
    test.append(SHO_interactions(5))
    test.append(Lennard_Jones(1,1))
    
    test = 1
    configuration = {
        'kB' : 1.0, # put as a constant 
        'Temperature' : 1.0,
        'N' : 10,
        'DIM' : 1,
        'm' : 1,
        'hamiltonian' : test,
        }
    
    integration_setting = {
        'iterations' : 10000,
        'DumpFreq' : 1,
        'dq' : 0.1,
        'seed' : 9654645,
        }
    
    configuration.update(integration_setting)
    MCMC = MSMC(**configuration)
    
    q_list = MCMC.integrate()
    confStat.plot_stat(q_list, np.zeros(q_list.shape), 'q_dist', **configuration)
    
