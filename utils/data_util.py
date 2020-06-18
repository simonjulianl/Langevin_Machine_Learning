#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 20:59:47 2020

@author: simon
"""

import numpy as np
from collections import defaultdict
from itertools import product
import random

class data_loader:
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
                eg. q_N1000_T1_DIM1.npy 
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
        file_format = path + 'phase_space_N' + str(_total_particle) + '_T{}_DIM{}.npy'
            # by default each file has 10000 samples
        
        if samples > _total_particle :
            raise Exception('Samples exceeding {} is not allowed'.format(_total_particle))
            
        q_list = None 
        p_list = None
       
        for i, temp in enumerate(temperature) : 
            import math # check whether temperature is fractional
            fraction = math.modf(temp)[0] != 0 # boolean
            temp = str(temp).replace('.','-') if fraction else str(int(temp))
            #for fractional temp, use - instead of . when saving
            phase_space = np.load(file_format.format(temp,DIM))
            curr_q, curr_p = phase_space[0][:samples], phase_space[1][:samples]
            # truncate according to samples
            if i == 0 : # first iteration, copy the data 
                q_list = curr_q
                p_list = curr_p
            else :
                q_list = np.concatenate((q_list, curr_q))
                p_list = np.concatenate((p_list, curr_p))
            
            assert q_list.shape == p_list.shape # shape of p and q must be the same 
    
        return (q_list, p_list)
    
    
    @staticmethod
    def grid_split_data(q_list, p_list, ratio : float,  seed = 937162211):
        '''
        Using the proposed grid splitting method on the phase space 
        to split the q,p into grids into with the stipulated ratio 
        of ratio : (1- ratio) data strictly from DIM = 1, generalizing to upper dimensions
        would be ambiguous 

        Parameters
        ----------
        q_list : np.array of N X DIM 
            position array
        p_list : np.array of N X DIM
            momentum array
        ratio : float
            the ratio between the 2 splitting, for 0 < r < 1
        seed : float
            any number for random seed splitting ,
            default : 937162211 which is 9 digit prime number
            
        Returns
        -------
        coordinates : tuple
            first element being : 
                list of coordinates, (q, p) for the first grid of ratio r of N X 2 X DIM
            second elemtn being : 
                list of coordinates (q,p) for ratio (1-r) of shape N X 2 X DIM

        '''
        
        if not 0 < ratio < 1:
            raise Exception('ratio must strictly between 0 and 1')
            
        random.seed(seed) # set the seed 
        np.random.seed(seed)
        
        assert q_list.shape == p_list.shape
        N, DIM = q_list.shape
        if DIM != 1 : 
            raise Exception('DIM != 1 is not supported')
                
        def nested_dict(n, type):
            if n == 1:
                return defaultdict(type)
            else:
                return defaultdict(lambda : nested_dict(n-1,type))
            
        qtick = np.arange(np.min(q_list) -0.5 ,np.max(q_list) + 0.5,0.5)
        ptick = np.arange(np.min(p_list) - 0.5, np.max(p_list) + 0.5,0.5)
        # put excess so that all particles are included
        grid = nested_dict(2,list) # initialize the nested dict
        
        print('Total Grid : {}'.format(len(qtick) * len(ptick))) # mention total grids
        
        for x in qtick:
            for y in ptick:
                grid[x][y] = []
                
        qlist = list(grid.keys())
        plist = list(ptick)
     
        for position, momentum in zip(q_list,p_list):
            #do a linear search to find the correct grid
            lower_bound_pos, lower_bound_momentum = None, None
            i = 1 
            while lower_bound_pos is None:
                if position.item() < qlist[i]:
                    # print(qlist[i])
                    lower_bound_pos = qlist[i-1]
                i += 1
                
            i = 1 # reset counter
            while lower_bound_momentum is None:
                if momentum.item() < plist[i] :
                    lower_bound_momentum = plist[i-1]
                i += 1
            grid[lower_bound_pos][lower_bound_momentum].append((position,momentum))
            
        total_particle = 0
        for x in qtick:
            for y in ptick:
                total_particle += len(grid[x][y])
                
        #randomly choose the grid until N ~ ratio  and 1 - ratio 
        combination = list(product(qlist,plist))
        for i in range(10): # shuffle the all the grids, arbitrary choice 10 shuffling
            combination = random.sample(combination,len(combination)) 
            
        grid_first, grid_second = [], [] #coordinate of grids for training
        N_first, N_first_current = ratio * N,0
        
        i = 0
        while N_first_current < N_first:
            grid_first.append(combination[i])
            q,p = combination[i]
            N_first_current += len(grid[q][p])
            i += 1 
        
        
        grid_second = combination[i:]
        
        print('Actual Split : {:.4f}% first split / {:.4f}% second split '.format(
            100.*N_first_current / N, 100. * (total_particle - N_first_current) / N
            )) # mention the splitting 
    
        #populate the init_pos, init vel from grid coordinates 
        grids = [grid_first, grid_second]
        coordinates = []
        for i,each_grid in enumerate(grids) : 
            init_pos = []
            init_vel = []
            for q,p in each_grid:
                temporary = grid[q][p]
                for item in temporary : 
                    init_pos.append(item[0])
                    init_vel.append(item[1])
            init_pos = np.expand_dims(np.array(init_pos), axis = 1)
            init_vel = np.expand_dims(np.array(init_vel), axis = 1) #  N X 1 X  DIM 
            coordinates.append(np.concatenate((init_pos, init_vel), axis = 1)) # N X 2 XDIM
            
        return coordinates
        
    
                