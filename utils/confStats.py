#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 16:50:15 2020

@author: simon
"""

import numpy as np
import warnings

class confStat:
    '''Helper Class to get the statistic of the configuration
        
    Configurations must always in pos/vel matrix of N X DIM dimensions
    
    '''
    @staticmethod
    def temp(**configuration):
        '''
        Helper Function to obtain the kinetic energy based on the momentum of particles
        
        Parameters
        ----------
 
        **kwargs :  configuration state consisting 
            - vel : N X DIM matrix 
                Velocity matrix of the configuration of N X DIM shape
            - N : int
                total number of particles
            - DIM : int
                Dimension of the particles 
            - m : float 
                mass of the particle 
            - BoxSize : float 
                scaling of the box cell 
        Returns
        -------
        temperature : float
            The translational temperature of the configurations

        '''
        try : 
            N = configuration['N']
            DIM = configuration['DIM']
            m = configuration['m']
            vel = configuration['vel']
        except : 
            raise Exception('N / Dimension / Mass not supplied')
            
        try : 
            BoxSize = configuration['BoxSize']
        except :
            BoxSize = 1 
            warnings.warn('BoxSize not supplied, set to 1')
            
        ene_kin = 0.0 
        
        for i in range(N):
            real_vel = BoxSize * vel[i,:] # rescale each velocity according to the box size
            ene_kin += 0.5 * m * np.dot(real_vel,real_vel) # 1/2 m v^2 for constant mass
    
        ene_kin_aver = 1.0 * ene_kin / N
        temperature = 2.0 / DIM * ene_kin_aver # by equipartition theorem
        
        return temperature 
    
    @staticmethod
    def potential_energy( **configuration):
        '''
        Helper Function to get the energy of the state depending on the external potential energy
        
        Parameters
        ----------
        potential : string
            string representation of the potential U(q) where q is the position
            for eg. (q**2 - 1) ** 2.0 + q 
            
            the potential is assumed to be symmetrical around x, y and z axis
            
        **configuration : configuration state consisting 
            - N : int
                total number of particles
            - DIM : int
                Dimension of the particles 
            - BoxSize : float 
                scaling of the box cell 
            - potential : string
                string expression of the potential energy in U(q) 
                where Ux Uy and Uz are assumed to be symmetrical

        Returns
        -------
        Average Potential Energy : float
            Potential Energy must be induced due to external potential
        '''
        try : 
            N = configuration['N']
            DIM = configuration['DIM']
            pos = configuration['pos']
            potential = configuration['potential']
        except : 
            raise Exception('N / Dimension / Mass / Potential not supplied')
            
        try : 
            BoxSize = configuration['BoxSize']
        except :
            BoxSize = 1 
            warnings.warn('BoxSize not supplied, set to 1')
            
        ene_pot = 0.0 
        for i in range(N):
            real_pos = BoxSize * pos[i,:] # rescale each velocity according to the box size
            for j in range(DIM):  # The potential is split into Ux , Uy, Uz
                q = real_pos[j]
                ene_pot += eval(potential) 
    
        ene_pot_aver = 1.0 * ene_pot / N
        
        return ene_pot_aver
    
    @staticmethod
    def kinetic_energy(**configuration):
        '''
        Helper Function to obtain the translational kinetic energy
        
        Parameters
        ----------
 
        **kwargs :  configuration state consisting 
            - vel : N X DIM matrix 
                Velocity matrix of the configuration of N X DIM shape
            - N : int
                total number of particles
            - DIM : int
                Dimension of the particles 
            - m : float 
                mass of the particle 
            - BoxSize : float 
                scaling of the box cell 
        Returns
        -------
        Kinetic Energy : float
            The average translational kinetic energy of the configuration

        '''
        try : 
            N = configuration['N']
            m = configuration['m']
            vel = configuration['vel']
        except : 
            raise Exception('N / Dimension / Mass not supplied')
            
        try : 
            BoxSize = configuration['BoxSize']
        except :
            BoxSize = 1 
            warnings.warn('BoxSize not supplied, set to 1')
            
        ene_kin = 0.0 
        
        for i in range(N):
            real_vel = BoxSize * vel[i,:] # rescale each velocity according to the box size
            ene_kin += 0.5 * m * np.dot(real_vel,real_vel) # 1/2 m v^2 for constant mass
    
        ene_kin_aver = 1.0 * ene_kin / N
        
        return ene_kin_aver 
    
    @staticmethod 
    def force(**configuration):
        '''
        Static function to get the force given the current state and potential

        Parameters
        ----------
        **configuration : configuration state consisting
            - N : int
                total number of particles
            - DIM : int
                Dimension of the particles 
            - BoxSize : float 
                scaling of the box cell 
            - force : string
                string expression of the force dUdq
                where Ux Uy and Uz are assumed to be symmetrical

        Returns
        -------
        force : np.array (N X DIM shape)
            force matrix based on external potential of current state

        '''
        try : 
            N = configuration['N']
            DIM = configuration['DIM']
            pos = configuration['pos']
            dUdq = configuration['force']
        except : 
            raise Exception('N / Dimension / Mass / Potential not supplied')
            
        try : 
            BoxSize = configuration['BoxSize']
        except :
            BoxSize = 1 
            warnings.warn('BoxSize not supplied, set to 1')
            

        force = np.zeros((N,DIM)) # placeholder for force
        for i in range(N):
            real_pos = BoxSize * pos[i,:] # rescale each velocity according to the box size
            for j in range(DIM):  # The potential is split into Ux , Uy, Uz
                q = real_pos[j] # q is used to eval force expression
                force[i][j] = -eval(dUdq) # since F = -dU/dq
    
        return force