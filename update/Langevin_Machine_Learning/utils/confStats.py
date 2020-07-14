#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 16:50:15 2020

@author: simon
"""

import numpy as np
import warnings
import matplotlib.pyplot as plt 
from ..phase_space import phase_space

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
            particle = configuration['particle']
            DIM = configuration['DIM']
            m = configuration['m']
            vel = configuration['phase_space'].get_p() / m # v = p/m
        except : 
            raise Exception('N / Dimension / Mass / vel not supplied')
            
        try : 
            BoxSize = configuration['BoxSize']
        except :
            BoxSize = 1 
            warnings.warn('BoxSize not supplied, set to 1')
            

        #print('confStats.py temp vel',vel)
        ene_kin = []
        for i in range(N):
            ene_kin_ = 0.0
            #print('confStats.py vel[i,:]',vel[i,:])
            real_vel = BoxSize * vel[i,:] # rescale each velocity according to the box size
            #print('confStats.py real_vel',real_vel)
            #print('confStats.py real_vel*real_vel',np.multiply(real_vel,real_vel))

            for j in range(particle):
                #print('confStats.py sum real_vel[j, :]', real_vel[j, :] )
                #print('confStats.py sum', np.sum(np.multiply(real_vel[j,:], real_vel[j,:]), axis=0))
                ene_kin_ += 0.5 * m * np.sum(np.multiply(real_vel[j,:],real_vel[j,:]),axis=0) # 1/2 m v^2 for constant mass
                #print('confStats.py ene_kin',ene_kin_)

            ene_kin.append(ene_kin_)

        ene_kin = np.array(ene_kin)
        #print('confStats.py temp ene_kin',ene_kin)

        ene_kin_aver = 1.0 * ene_kin / particle
        #print('confStats.py temp ene_kin_aver',ene_kin_aver)
        temperature = 2.0 / DIM * ene_kin_aver # by equipartition theorem
        #print('confStats.py temp temperature',temperature)

        return temperature
    
    @staticmethod
    def kinetic_energy(**configuration):
        '''
        Helper Function to obtain the translational kinetic energy
        
        Parameters
        ----------
     
        **kwargs :  configuration state consisting 
            - N : int
                total number of particles
            - DIM : int
                Dimension of the particles 
        Returns
        -------
        ene_kin_aver : float
            The average translational kinetic energy of the configuration
    
        '''
        try : 
            DIM = configuration['DIM']
        except : 
            raise Exception('particle / DIM ot supplied')

        #print('confStats.py **configuration',configuration)
        ene_kin_aver = confStat.temp(**configuration) * DIM / 2 
        #print('confStats.py kinetic_energy',ene_kin_aver)
        return ene_kin_aver
    
    @staticmethod
    def plot_stat(q_hist : list , p_hist : list , mode : str, **configuration):
        '''
        Static function to help plot various statistic according to the supplied
        trajectories of qlist and plist as well as p 
        
        Parameters
        ----------
        qlist : np.array 
            qlist must be in shape of samples x N X DIM, if not please resize
        plist : np.array
            plist must be in shape of samples x N X DIM, if not please resize 
        mode : str
            various modes available : energy, p, potential, kinetic ,q , v_dist, q_dist, speed_dist
        **configuration : configuration of the state
            kB : float
                boltzmann constant
            Temperature : float
                temperature of the state 

        Raises
        ------
        Exception
            Error in Modes supplied or kB/ Temperature not supplied in configuration

        '''
        line_plot = ['energy', 'p', 'potential', 'kinetic','q','all']
        hist_plot = ['v_dist', 'q_dist', 'speed_dist']
 
        if mode not in line_plot and mode not in hist_plot:
            raise Exception('Modes not available , check the mode')
            
        assert q_hist.shape == p_hist.shape # q list and p list must have the same size
        
        color = {
            'p' : 'blue',
            'q' : 'blue',
            'potential' : 'orange',
            'kinetic' : 'orange',
            'energy' : 'orange',
            'q_dist' : 'black',
            'v_dist' : 'black',
            'speed_dist' : 'black'
            }
        
        dim = {0 : 'x', 1 : 'y', 2 :'z'}
        hamiltonian = configuration['hamiltonian']
        BoxSize = configuration['BoxSize']
        periodicity = configuration['periodicity']

        #print('confStats.py hamiltonian',hamiltonian)
        if mode in line_plot :      
            potential = []
            kinetic = []
            #print('confStats.py q_hist',q_hist.shape)
            #print('confStats.py p_hist', p_hist.shape)
            for i in range(len(q_hist)):
                p_dummy_list = np.zeros(q_hist[i].shape)
                temporary_phase_space = phase_space()
                temporary_phase_space.set_q(q_hist[i])
                #print('confStats.py q_hist', q_hist[i])
                temporary_phase_space.set_p(p_dummy_list)

                #print('confStats.py temporary_phase_space ',temporary_phase_space)
                potential.append(hamiltonian.total_energy(temporary_phase_space, BoxSize,periodicity=True)) # ADD periodicity=True
                #print('confStats.py potential',potential)
                configuration['phase_space'].set_p(p_hist[i])
                #print('confStats.py p_hist',p_hist[i])

                kinetic_ = confStat.kinetic_energy(**configuration)
                #print('confStats.py kinetic_',kinetic_)
                kinetic.append(kinetic_)

            kinetic = np.array(kinetic)
            potential = np.array(potential)
            energy = kinetic + potential

            #print('confStats.py kinetic', kinetic)
            #print('confStats.py potential',potential)
            #print('confStats.py energy',energy)
            #print(energy.shape[1])
            #print('confStats.py energy T',energy.T)

            if mode == 'p' or mode == 'q' : # for p and q we plot dimension per dimension
                for n in range(configuration['DIM']):
                    if mode == 'p':
                        plt.plot(p_hist[:,:,:,n], color = color[mode], label = 'p')
                    elif mode == 'q':
                        plt.plot(q_hist[:,:,:,n], color = color[mode], label = 'q')
                    plt.xlabel('sampled steps')
                    plt.ylabel(mode + ' ' + dim[n])
                    plt.legend(loc = 'best')
                    plt.show()

            elif mode == 'all':

                for i in range(energy.shape[1]):
                    if i < 2:
                        plt.plot(energy.T[i], label = 'total energy')
                        plt.plot(kinetic.T[i], label='kinetic energy')
                        plt.plot(potential.T[i], label='potential energy')

                        plt.xlabel('sampled steps')
                        plt.legend(loc = 'best')
                        plt.show()

            else :
                if mode == 'energy' : # if energy , we use average on every dimension
                    plt.plot(energy, color = color[mode], label = 'total energy')
                elif mode =='kinetic' :
                    plt.plot(kinetic, color = color[mode], label = 'kinetic energy')
                elif mode == 'potential' :
                    plt.plot(potential.T[0][100:], color = color[mode], label = 'potential energy')

                plt.xlabel('sampled steps')
                plt.ylabel(mode)
                plt.legend(loc = 'best')
                plt.show()


                    
        else : 
            try : 
                _beta = 1 / (configuration['kB'] * configuration['Temperature'])
                _m = configuration['m']
            except : 
                raise Exception('kB / Temperature not set ')
                
            for n in range(configuration['DIM']):
                if mode == 'q_dist':
                    print('confState.py q_hist',q_hist.shape)
                    print(q_hist[:,:,:,n].shape)
                    curr = q_hist[:,:,:,n].reshape(-1,2) # collapse to 1 long list
                    print(curr.shape)

                    #plot exact
                    q = np.linspace(np.min(curr),np.max(curr),1000)
                    #create hamiltonian list
                    potential = np.array([])
                    for i in range(len(q)):
                        q_list_temp = np.expand_dims(q[i], axis = 0).reshape(2,2)
                        print('confState.py q_list_temp',q_list_temp)

                        p_list_temp = np.zeros(q_list_temp.shape) # prevent KE from integrated
                        potential = np.append(potential,
                                              hamiltonian.total_energy(q_list_temp, p_list_temp))
                    prob_q = np.exp(-_beta * potential)
                    dq = q[1:] - q[:-1]
                    yqs = 0.5 * (prob_q[1:] + prob_q[:-1])
                    Zq = np.dot(yqs.T, dq) # total area
                    plt.plot(q,prob_q/Zq,marker = None, color = "red", linestyle = '-',label = 'q exact') 
                    
                elif mode == 'v_dist': 
                    curr = p_hist[:,:,n].reshape(-1,1) / _m # collapse to 1 long list
                    #plot exact
                    v = np.linspace(np.min(curr),np.max(curr),1000)
                    prob_v = ((_m * _beta)/ (2 * np.pi))**0.5 * np.exp(-_beta * (v ** 2.0) / 2)
                    
                    plt.plot(v,prob_v,marker = None, color = "red", linestyle = '-',label = 'v exact') 
                
                elif mode == 'speed_dist':
                    curr = p_hist[:,:,:].reshape(-1,configuration['DIM']) / _m # collapse to 1 long list of N X DIM
                    speed = np.linalg.norm(curr, 2, axis = 1)
                    v = np.linspace(np.min(speed),np.max(speed),1000)
                    prob_v = 4 * np.pi * (_beta/(2 * np.pi)) ** 1.5 * v ** 2.0 *  np.exp(-_beta * v ** 2.0 / (2 * _m))
                    plt.plot(v,prob_v,marker = None, color = "red", linestyle = '-',label = 'v exact') 
                    
                interval = (np.max(curr) - np.min(curr)) / 30
                values, edges = np.histogram(curr, bins = np.arange(np.min(curr), np.max(curr) , interval),
                         density = True) # plot pdf 
                center_bins = 0.5 * (edges[1:] + edges[:-1])
                plt.plot(center_bins, values, color = color[mode] , label = mode)
                plt.ylabel('pdf')
                plt.legend(loc = 'best')
                
                
                if mode == 'speed_dist' : 
                    plt.xlabel(mode[0])
                    plt.show()
                    return #exit the function since we take all the dimensions at once
                
                plt.xlabel(mode[0] + dim[n])
                plt.show()
