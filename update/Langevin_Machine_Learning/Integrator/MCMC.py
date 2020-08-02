#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 15:52:06 2020

@author: simon
"""

import numpy as np
from tqdm import trange
import copy
import random
from ..utils.confStats import confStat 
from .base_simulation import Integration
from ..phase_space.phase_space import phase_space
from ..hamiltonian.pb import periodic_bc
import warnings

class MCMC(Integration):
    '''
    This is a Monte Carlo Molecular Simulation
    Only integrate potential function, sampling is for momentum is done exact 
    This class is only used to generate initial configurations
    '''
    
    def helper(self = None): 
        '''print the common parameters helper'''
        for parent in MCMC.__bases__:
            print(help(parent))
        
    def __init__(self, *args, **kwargs) -> object:
        ''' initialize the NVT MSMC module
        Temperature scaled for initialization 
        
        Parameters
        ----------
        
        **kwargs : integration setting 
        
        iterations : int
            total number of Monte Carlo Steps
            

        DumpFreq : int
            Sampling frequency 
            default : 1 step ( Sample every step )
            
        dq : float 
            random walk of position for Monte Carlo
            
        ! user helper function        self.random_1 = random_1
        self.random_2 = random_2.reshape(-1,self.N)to get full parameter setting !
        
        Returns
        -------
        MSMC Class Object
        
        '''

        super(MCMC, self).__init__(*args, **kwargs)
                
        #Configuration settings 
        try : 
            self._intSetting = {
                'iterations' : kwargs['iterations'],
                'DumpFreq' : kwargs['DumpFreq'],
                'dq' : kwargs['dq'],
                }
            
            if not self._intSetting['iterations'] >= self._intSetting['DumpFreq'] :
                raise ValueError('DumpFreq must be smaller than iterations')

        except : 
            raise TypeError('Integration setting error ( iterations / DumpFreq / dq  )')
            
        # seed setting
        try :
            seed = kwargs['seed']
            np.random.seed(int(seed))
            random.seed(int(seed))
        except :
            warnings.warn('Seed not set, start using default numpy/random/torch seed')
            
        # # temperature scaling
        #
        # Temp = confStat.temp(**self._configuration) # get current temperature
        # scalefactor = np.sqrt(self._configuration['Temperature']/Temp)
        #
        # #lambda constant of (Tn / T0) ^0.5 by adjusting KE where Tn is target temp, T0 is initial temp
        #
        # curr_vel = self._configuration['phase_space'].get_p() / kwargs['m']
        # self._configuration['phase_space'].set_p(np.multiply(curr_vel , scalefactor))
    
    def mcmove(self) :
        '''
        Helper function for a Monte Carlo Integration 
        Random Walk with the proposed distribution of U[-dq,dq)
        
        Only integrate potential and not kinetic 

        Returns
        -------
        None
            directly change the current configuration setting

        '''
        curr_q = self._configuration['phase_space'].get_q()
        print('MCMC.py curr_q', curr_q)
        eno_q = self._configuration['hamiltonian'].total_energy(phase_space,self._configuration['pb_q'])

        print('MCMC.py eno_q', eno_q)
        trial = random.randint(0, len(curr_q) - 1) # randomly pick one particle from the state
        print('MCMC.py trial', trial)
        #eno is the old potential energy configuration
        q_list = self._configuration['phase_space'].get_q()
        print('MCMC.py q_list.shape', q_list)
        print('MCMC.py curr_q[trial]',np.array(curr_q[trial]))

        #perform random step with proposed uniform distribution
        qn = np.array(curr_q[trial]) + (np.random.uniform(0,1, np.array(curr_q[trial]).shape) - 0.5) * self._intSetting['dq']
        print('MCMC.py qn', qn)
        qo = np.array(curr_q[trial]) # store old position
        print('MCMC.py qo', qo)

        curr_q[trial] = qn # set new position
        print('MCMC.py curr_q[trial]',np.array(curr_q[trial]))
        print('MCMC.py curr_q ', np.array(curr_q))
        self._configuration['phase_space'].set_q(curr_q) # try the new state
            
        #enn is the new potential energy configuration
        q_list = self._configuration['phase_space'].get_q()
        self._configuration['pb_q'] = periodic_bc()
        quit()
        enn_q = self._configuration['hamiltonian'].total_energy(phase_space,self._configuration['pb_q'])
     
        #accept with probability proportional di e ^ -beta * delta E
        if random.uniform(0,1) >= np.exp(-self._configuration['beta'] * (enn_q - eno_q)):
            self.rejection += 1 # rejected
            curr_q[trial] = qo # restore the old position
            self._configuration['phase_space'].set_q(curr_q) 
        
        
    def integrate(self):
        '''
        Implementation of integration for Monte Carlo Integration 
        
        Raises
        ------
        Exception
            Failed MC Move , check self.mcmove() function

        Returns
        -------
        q_list : np.array ( Total Sample X N X DIM )
            array of list of sampled q obtained

        '''
        self.rejection = 0 # reset the rejection counter to check the integration stats 
        total_samples = self._intSetting['iterations'] // self._intSetting['DumpFreq']
        q_list = np.zeros((total_samples,self._configuration['particle'],self._configuration['DIM']))
        print('MCMC.py q_list',q_list.shape)
        for i in trange(1, self._intSetting['iterations'] + 1, desc = "simulating"):
            self.mcmove()
            if i % self._intSetting['DumpFreq'] == 0 :
                q_list[i-1] = copy.deepcopy(self._configuration['phase_space'].get_q())
                
        
        print('Rejection Rate : {}%'.format(self.rejection * 100/ self._intSetting['iterations']))
        #print out the rejection rate, recommended rejection 40 - 60 % based on Lit
        
        return q_list
    
    def __repr__(self):
        state = super().__repr__() 
        state += '\nIntegration Setting : \n'
        for key, value in self._intSetting.items():
            state += str(key) + ': ' +  str(value) + '\n' 
        return state

