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
from random  import seed
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
                'DISCARD' : kwargs['DISCARD'],
                'dq' : kwargs['dq'],
                }
            
            if not self._intSetting['iterations'] >= self._intSetting['DISCARD'] :
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
        #print('MCMC.py _configuration',self._configuration)
        curr_q = self._configuration['phase_space'].get_q()
        #print('MCMC.py curr_q', curr_q)
        #print('MCMC.py curr_q.shape', curr_q.shape)
        self.eno_q = self._configuration['hamiltonian'].total_energy(self._configuration['phase_space'],self._configuration['pb_q'])

        #print('MCMC.py eno_q', eno_q)
        #print('MCMC.py curr_q.shape',curr_q.shape)
        #print('MCMC.py len(curr_q)',curr_q.shape[1])
        trial = random.randint(0, curr_q.shape[1]-1) # randomly pick one particle from the state
        #print('MCMC.py trial', trial)

        #print('MCMC.py curr_q', curr_q)
        #print('MCMC.py curr_q[0]', curr_q[:,0])
        #print('MCMC.py curr_q[1]', curr_q[:,1])
        #print('MCMC.py curr_q[trial]', curr_q[:,trial])

        old_q = np.copy(curr_q[:,trial])
        #perform random step with proposed uniform distribution
        curr_q[:,trial] = old_q + (np.random.rand(1,self._configuration['DIM'])-0.5)* self._intSetting['dq']

        #print('MCMC.py curr_q ', curr_q)
        self._configuration['pb_q'].adjust_real(curr_q,self._configuration['BoxSize'])
        #print('MCMC.py curr_q pbc', curr_q)
        self._configuration['phase_space'].set_q(curr_q)

        self.enn_q = self._configuration['hamiltonian'].total_energy(self._configuration['phase_space'],self._configuration['pb_q'])
        #print('MCMC.py enn_q', enn_q)

        dU = self.enn_q - self.eno_q
        #print('dU',dU)
        #print('accept curr_q',curr_q)
        self._configuration['phase_space'].set_q(curr_q)
        #accept with probability proportional di e ^ -beta * delta E
        self.ACCsum += 1.0
        self.ACCNsum += 1.0
        if (dU > 0):
            if (np.random.rand() > np.exp(-self._configuration['beta'] * dU)):
                self.ACCsum -= 1.0 # rejected
                curr_q[:,trial] = old_q # restore the old position
                #print('reject curr_q',curr_q)
                self.enn_q = self.eno_q
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
        self.ACCsum = 0.
        self.ACCNsum = 0.
        # specific heat calc
        TE1sum = 0.0
        TE2sum = 0.0
        Nsum = 0.0
        total_samples = self._intSetting['iterations'] - self._intSetting['DISCARD']
        q_list = np.zeros((total_samples,self._configuration['particle'],self._configuration['DIM']))
        U = np.zeros(total_samples)

        #print('MCMC.py integrate q_list',q_list)
        for i in trange(0, self._intSetting['iterations'], desc = "simulating"):
            #print('iteration',i)
            for _ in range(self._configuration['DIM']):
                self.mcmove()
            #print('ACC each mcs' , self.ACCsum)
            if(i >= self._intSetting['DISCARD']):
                q_list[i-self._intSetting['DISCARD']] = copy.deepcopy(self._configuration['phase_space'].get_q())
                U[i-self._intSetting['DISCARD']] = self.enn_q
                TE1sum += self.enn_q
                TE2sum += (self.enn_q * self.enn_q)
                Nsum += 1.0

        spec = (TE2sum / Nsum - TE1sum * TE1sum / Nsum / Nsum) / self._configuration['Temperature'] / self._configuration['Temperature']  / self._configuration['particle']
        print('ACC Rate : {}%'.format(self.ACCsum/ self.ACCNsum))
        print('spec',spec)
        #print out the rejection rate, recommended rejection 40 - 60 % based on Lit
        
        return q_list, U, self.ACCsum/ self.ACCNsum, spec
    
    def __repr__(self):
        state = super().__repr__() 
        state += '\nIntegration Setting : \n'
        for key, value in self._intSetting.items():
            state += str(key) + ': ' +  str(value) + '\n' 
        return state

