#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 17:30:43 2020

@author: simon
"""

import numpy as np
from abc import ABC, abstractmethod 
import warnings 
from ..hamiltonian.hamiltonian import Hamiltonian
from ..phase_space.phase_space import phase_space
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
            -pos : np.array ( N X DIM Shape ) (Optional)
                Position Matrix ( q )
            -vel : np.array( N X DIM Shape ) (Optional)
                Velocity Matrix ( p ) 
            the potential is assumed to be symmetrical around x, y and z axis
        
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
                'hamiltonian' : hamiltonian,
            }
        except : 
            raise ValueError('N / DIM / m / hamiltonian unset or error')

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
            
        pos = kwargs.get('pos', None)
        vel = kwargs.get('vel', None)
        
        if pos is None :  # create random particle initialization of U[-1,1) for q (pos) and v if not supplied
            pos = np.random.uniform(-1, 1, (self._configuration['N'], self._configuration['DIM']))
            MassCentre = np.sum(pos,axis = 0) / self._configuration['N']
            for i in range(self._configuration['DIM']) :
                pos[:,i] = pos[:,i] - MassCentre[i]
        
        if vel is None :
            vel = np.random.uniform(-1, 1, (self._configuration['N'], self._configuration['DIM']))
            if self._configuration['N'] == 1 : 
                warnings.warn('Initial velocity and pos is not adjusted to COM and external force')
            else :
                VelCentre = np.sum(vel, axis = 0) / self._configuration['N']
                for i in range(self._configuration['DIM']):
                    vel[:,i] = vel[:,i] - VelCentre[i]
                
        self._configuration['phase_space'] = phase_space()
        self._configuration['phase_space'].set_q(pos)
        self._configuration['phase_space'].set_p(vel * kwargs['m'])
        
        self._configuration['periodicity'] = bool(kwargs.get('periodicity', False)) # cast to boolean
        
        if self._configuration['periodicity']:  # applying boundary condition 
            try :
                self._configuration['BoxSize'] = kwargs['BoxSize'] # get boxsize if possible , by default one 
                #only PBC is considered here, rigid boundary condition and others are not considered.
            except :
                raise Exception('periodicity is True, but BoxSize / boundary not found')
            
    @abstractmethod
    def integrate(self):
        pass
    
    def _filename_creator(self): 
        '''helper function to create the filename based on current configuration'''
        
        base_library = os.path.abspath('Langevin_Machine_Learning/init')
        N = self._configuration['N']
        temp = self._configuration['Temperature']
        DIM = self._configuration['DIM']
        
        import math #check whether temperature is fractional
        fraction = math.modf(temp)[0] != 0 # boolean
        temp = str(temp).replace('.','-') if fraction else str(int(temp)) # tokenizer, change . to -
            
        filename = '/phase_space_N{}_T{}_DIM{}.npy'.format(N, temp, DIM)
        return base_library + filename
    
    def set_phase_space(self, samples : int = -1) :
        '''
        Base Loader function for load p and q given MCMC initialization
        all the numpy file must be saved in the init folder 
        
        Parameters
        ----------
        samples : int, optional
            how many sampels per temperature stated in the configuration 
            by default -1, meaning take everything in the init

        '''
        file_path = self._filename_creator()
        self._configuration['phase_space'].read(file_path, samples)
        
        if self._configuration['periodicity']: 
            phase_space = self._configuration['phase_space']
            scaled_q = phase_space.get_q() / self._configuration['BoxSize']
            scaled_p = phase_space.get_p() / self._configuration['BoxSize']
            phase_space.set_q(scaled_q)
            phase_space.set_p(scaled_p) # scale to box size of -1 ro 1
            
        q_list = self._configuration['phase_space'].get_q() #sample the shape and DIM
        
        if samples > self._configuration['N']:
            raise Exception('samples exceed available particles')
            
        # the loaded file must contain N X DIM matrix of p and q
        _new_N, _new_DIM = q_list.shape
       
        if self._configuration['N'] != _new_N : 
            warnings.warn('N is changed to the new value of  {}'.format(_new_N))
            self._configuration['N'] = _new_N
            
        if self._configuration['DIM'] != _new_DIM : 
            warnings.warn('DIM is changed to the new value of  {}'.format(_new_DIM))
            self._configuration['DIM'] = _new_N
                
    def save_phase_space(self, filename = None): 
        '''if None, by default its save in init Langevin_Machine_Learning/init
        wrapper function to save the phase space onto the init file
        
        Parameters
        ----------
        filename : str, optional
            default is None. Meaning save in init folder, can be modified to other folders 
        '''
        phase_space = self._configuration['phase_space']
        
        if self._configuration['periodicity'] :
            scaled_q = phase_space.get_q() * self._configuration['BoxSize']
            scaled_p = phase_space.get_p() * self._configuration['BoxSize']
            phase_space.set_q(scaled_q)
            phase_space.set_p(scaled_p)
            
        file_path  = self._filename_creator()
        phase_space.write(filename = file_path)
        
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
