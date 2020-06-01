#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 20:59:47 2020

@author: simon
"""

import matplotlib.pyplot as plt
import numpy as np
from .confStats import confStat

class data_util:
    '''
    Class of helper functions to plot various statistics 
    '''
    @staticmethod 
    def loadp_q(path : str, temperature : list , samples : int, DIM : int):
        '''
        Function to load p and q based on available files
        N and DIM will be adjusted to loaded p and q 
        Parameters
        ----------
        path : str
            path to initialization folder
            
            default file name : 
                eg. q_N1000_T1_DIM1_MCMC.npy 
                means the file contains q for 10 000 samples at T = 1, kB is assumed to be 1
                at DIM = 1
                
        temperature : list
            list of temperature to be loaded
            
        samples : int
            total samples per temperature included
            
        DIM : int
            dimension of the particles to be loaded
            
        Raises
        ------
        Exception
            Failed Initialization, unable to find file / load file
        
        Precaution
        ----------
        For Fractional temp, please save the file using - instead of . for decimals
            
        Return 
        ------
        q_list : np.array
            array of loaded q_list of N X DIM matrix
            
        p_list : np.array
            array of loaded p_list of N X DIM matrix 
        '''
        
        import os 
        
        if not os.path.exists(path) :
            raise Exception('path doesnt exist')
            
        _total_particle = 2500
        file_format = path + '{}_N' + str(_total_particle) + '_T{}_DIM{}_MCMC.npy'
            # by default each file has 10000 samples
        
        if samples > _total_particle :
            raise Exception('Samples exceeding {} is not allowed'.format(_total_particle))
            
        q_list = None 
        p_list = None
       
        for temp in temperature : 
            import math # check whether temperature is fractional
            fraction = math.modf(temp)[0] != 0 # boolean
            temp = str(temp).replace('.','-') if fraction else str(int(temp))
            #for fractional temp, use - instead of . when saving
            curr_q = np.load(file_format.format('q',temp,DIM))[:samples] # truncate according to samples
            curr_p = np.load(file_format.format('p',temp,DIM))[:samples]
            if q_list == None or p_list == None : 
                q_list = curr_q
                p_list = curr_p
            else :
                q_list = np.concatenate((q_list, curr_q))
                p_list = np.concatenate((p_list, curr_p))
            
            assert q_list.shape == p_list.shape # shape of p and q must be the same 
    
        return (q_list, p_list)
    
    @staticmethod
    def plot_stat(qlist : list , plist : list , mode : str, **configuration):
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
            various modes available : energy, p, potential, kinetic ,q , p_dist, q_dist
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
        line_plot = ['energy', 'p', 'potential', 'kinetic','q']
        hist_plot = ['p_dist', 'q_dist']
 
        if mode not in line_plot and mode not in hist_plot:
            raise Exception('Modes not available , check the mode')
            
        assert qlist.shape == plist.shape # q list and p list must have the same size
        
        color = {
            'p' : 'blue',
            'q' : 'blue',
            'potential' : 'orange',
            'kinetic' : 'orange',
            'energy' : 'orange',
            'q_dist' : 'black',
            'p_dist' : 'black'
            }
        
        dim = {0 : 'x', 1 : 'y', 2 :'z'}
        if mode in line_plot : 
            kinetic_energy = []
            potential_energy = []
            energy = []
            for i in range(len(qlist)):
                temp_conf = configuration # make dummy state to be passed 
                temp_conf['pos'] = qlist[i]
                temp_conf['vel'] = plist[i]
                kinetic_energy.append(confStat.potential_energy(**temp_conf))
                potential_energy.append(confStat.kinetic_energy(**temp_conf))
            
            energy = np.array(kinetic_energy) + np.array(potential_energy)
            
            if mode == 'p' or mode == 'q' : # for p and q we plot dimension per dimension
                for n in range(configuration['DIM']):
                    if mode == 'p':
                        plt.plot(plist[:,:,n], color = color[mode], label = 'p')
                    elif mode == 'q':
                        plt.plot(qlist[:,:,n], color = color[mode], label = 'q')
                    plt.xlabel('sampled steps')
                    plt.ylabel(mode + ' ' + dim[n])
                    plt.legend(loc = 'best')
                    plt.show()
            else : 
                if mode == 'energy' : # if energy , we use average on every dimension
                    plt.plot(energy, color = color[mode], label = 'total energy')
                elif mode =='kinetic' : 
                    plt.plot(kinetic_energy, color = color[mode], label = 'kinetic energy')
                elif mode == 'potential' : 
                    plt.plot(potential_energy, color = color[mode], label = 'potential energy')
                
                plt.xlabel('sampled steps')
                plt.ylabel(mode)
                plt.legend(loc = 'best')
                plt.show()
                    
        else : 
            try : 
                _beta = 1 / (configuration['kB'] * configuration['Temperature'])
            except : 
                raise Exception('kB / Temperature not set ')
                
            for n in range(configuration['DIM']):
                if mode == 'q_dist':
                    curr = qlist[:,:,n].reshape(-1,1) # collapse to 1 long list
                    #plot exact
                    q = np.linspace(np.min(curr),np.max(curr),1000)
                    prob_q = eval("np.exp(-_beta * {})".format(configuration['potential']))
                    
                    dq = q[1:] - q[:-1]
                    yqs = 0.5 * (prob_q[1:] + prob_q[:-1])
                    Zq = np.dot(yqs.T, dq) # total area
                    plt.plot(q,prob_q/Zq,marker = None, color = "red", linestyle = '-',label = 'q exact') 
                    
                elif mode == 'p_dist': 
                    curr = plist[:,:,n].reshape(-1,1) # collapse to 1 long list
                    #plot exact
                    p = np.linspace(np.min(curr),np.max(curr),1000)
                    prob_p = np.exp(-_beta * (p ** 2.0) / 2)
                    
                    dp = p[1:] - p[:-1]
                    yps = 0.5 * (prob_p[1:] + prob_p[:-1])
                    Zp = np.dot(yps.T, dp) # total area
                    print(Zp)
                    plt.plot(p,prob_p/Zp,marker = None, color = "red", linestyle = '-',label = 'p exact') 
                    
                interval = (np.max(curr) - np.min(curr)) / 30
                values, edges = np.histogram(curr, bins = np.arange(np.min(curr), np.max(curr) , interval),
                         density = True) # plot pdf 
                center_bins = 0.5 * (edges[1:] + edges[:-1])
                plt.plot(center_bins, values, color = color[mode] , label = mode)
                plt.xlabel(mode[0] + dim[n])
                plt.ylabel('pdf')
                plt.legend(loc = 'best')
                plt.show()
    
    @staticmethod        
    def plot_loss(loss : list, mode : str) :
        '''
        helper function to plot loss

        Parameters
        ----------
        loss : list
            np.array of loss 
        mode : str
            change the label for validation, train, test modes

        Raises
        ------
        Exception
            modes not found
        '''
        if mode not in ['validation', 'train', 'test']:
            raise Exception('mode not found, please check the mode')
            
        if mode == 'validation':
            plt.plot(loss, color = 'blue', label='validation loss')
        elif mode =='train' : 
            plt.plot(loss, color = 'blue', label = 'train loss')
        else : 
            plt.plot(loss, color = 'blue', label = 'test loss')
            
        plt.legend(loc = 'best')
        plt.xlabel('epoch')
        plt.show()       
                