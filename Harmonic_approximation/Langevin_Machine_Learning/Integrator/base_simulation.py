#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 17:30:43 2020

@author: simon
"""
from typing import Dict, Any, Union

import numpy as np
from abc import ABC, abstractmethod 
import warnings 
from ..hamiltonian.hamiltonian import Hamiltonian
from ..phase_space.phase_space import phase_space
from ..hamiltonian.pb import periodic_bc
from ..utils import confStat
import os

class Integration(ABC) : 

    @abstractmethod
    def __init__(self, *args, **kwargs) -> object:           
        ''' initialize a configuration state 
        Parameters
        ----------            
        **kwargs : 
            - N : int
                total number of particles
                default : 1
            - DIM : int
                Dimension of the particles 
                default : 1 Dimension
            - m : float 
                mass of the particle 
                default : 1
            - kB : float
                boltzmann constant
                default : 1 unit
            - Temperature : float
                Temperature at which the MC Steps are conducted
                default : 1 unit
            -hamiltonian : Hamiltonian
                Hamiltonian class consisting all the interactions
                
            the potential is assumed to be symmetrical around x, y and z axis
                
        Position and Velocity Matrix : N x DIM matrix 
        
        Returns
        -------
        Abstract Base Class, cannot be instantiated
        '''
        #Configuration settings 
        hamiltonian = kwargs['hamiltonian']

        if not isinstance(hamiltonian, Hamiltonian):
            raise Exception('not a Hamiltonian class ')

        try :
            self._configuration = {
                'N' : kwargs['N'],
                'DIM' : kwargs['DIM'],
                'm' : kwargs['m'],
                'particle' : kwargs['particle'], ### Add
                'hamiltonian' : hamiltonian,
                'BoxSize': kwargs['BoxSize']
            }
        except :
            raise ValueError('N / DIM / m / hamiltonian / Boxsize unset or error')

        #Constants
        try : 
            constants = {
                'Temperature' : kwargs['Temperature'],
                'kB' : kwargs['kB']
                }
            
            constants['beta'] = 1 / (constants['Temperature'] * constants['kB'])
    
            self._configuration.update(constants)
            
            del constants 
            
        except : 
            raise ValueError('Constant kB / Temperature Setting Error')

        # just create container
        pos = kwargs.get('pos', None)
        vel = kwargs.get('vel', None)
        
        if pos is None :  # create random particle initialization of U[-1,1) for q (pos) and v if not supplied
            pos = np.random.uniform(-0.5, 0.5, ( self._configuration['particle'], self._configuration['DIM'])) ### ADD particle
            pos = pos * self._configuration['BoxSize']
            pos = np.expand_dims(pos, axis=0)

        if vel is None :
            vel = []
            #'generate': 'maxwell'
            sigma = np.sqrt(self._configuration['Temperature']) # sqrt(kT/m)
            for i in range(self._configuration['N']):
                vx = np.random.normal(0.0,sigma,self._configuration['particle'])
                vy = np.random.normal(0.0,sigma,self._configuration['particle'])
                vel_xy = np.stack((vx,vy),axis=-1)
                vel.append(vel_xy)

            vel = np.array(vel)

        self._configuration['phase_space'] = phase_space()
        self._configuration['phase_space'].set_q(pos)
        self._configuration['phase_space'].set_p(vel * kwargs['m'])
        self._configuration['pb_q'] = periodic_bc()

        #Scaling velocity
        # Temp = confStat.temp(**self._configuration)
        # scalefactor = np.sqrt(self._configuration['Temperature']/Temp) # To make sure that temperature is exactly wanted temperature
        # scalefactor = scalefactor[:,np.newaxis,np.newaxis]
        # vel = vel*scalefactor
        #self._configuration['phase_space'].set_p(vel)

    @abstractmethod
    def integrate(self):
        pass
    
    def _filename_creator(self): 
        '''helper function to create the filename based on current configuration'''
        
        base_library = os.path.abspath('Langevin_Machine_Learning/init')

        return base_library
    
    def set_phase_space(self, nsamples : int = -1) :
        '''
        Base Loader function for load p and q given MCMC initialization
        all the numpy file must be saved in the init folder 
        
        Parameters
        ----------
        samples : int, optional
            how many sampels per temperature stated in the configuration 
            by default -1, meaning take everything in the init

        '''
        temp = self._configuration['Temperature']
        particle = self._configuration['particle']

        import math  # check whether temperature is fractional
        fraction = math.modf(temp)[0] != 0  # boolean
        filename = '/N_particle{}_samples{}_rho0.1_T{}_pos_sampled.npy'.format(particle,nsamples,temp)
        file_path = self._filename_creator() + filename
        self._configuration['phase_space'].read(file_path, nsamples)

        inital_q_list = self._configuration['phase_space'].get_q() #sample the shape and DIM
        inital_p_list = self._configuration['phase_space'].get_p()  # sample the shape and DIM

        if nsamples > self._configuration['N']:
            raise Exception('samples exceed available particles')
            
        # the loaded file must contain N X particle x DIM matrix of p and q
        _new_N, _new_particle, _new_DIM = inital_q_list.shape
       
        if self._configuration['N'] != _new_N : 
            warnings.warn('N is chanpged to the new value of  {}'.format(_new_N))
            self._configuration['N'] = _new_N

        if self._configuration['particle'] != _new_particle:
            warnings.warn('particle is changed to the new value of  {}'.format(_new_particle))
            self._configuration['particle'] = _new_particle

        if self._configuration['DIM'] != _new_DIM : 
            warnings.warn('DIM is changed to the new value of  {}'.format(_new_DIM))
            self._configuration['DIM'] = _new_DIM

        return (inital_q_list, inital_p_list)
                
    def save_phase_space(self,q_hist: list , p_hist: list ,filename = None):
        '''if None, by default its save in init Langevin_Machine_Learning/init
        wrapper function to save the phase space onto the init file
        
        Parameters
        ----------
        filename : str, optional
            default is None. Meaning save in init folder, can be modified to other folders 
        '''
        phase_space_ = np.array((q_hist, p_hist))
        file_path  = self._filename_creator() + filename
        np.save(file_path, phase_space_)
        
    def get_configuration(self):
        '''getter function for configuration'''
        return self._configuration
    
    def __repr__(self):
        state = 'current state : \n'
        for key, value in self._configuration.items():
            if key == 'hamiltonian' :
                state += key + ':\n' + value.__repr__()
            state += str(key) + ': ' +  str(value) + '\n' 
        return state