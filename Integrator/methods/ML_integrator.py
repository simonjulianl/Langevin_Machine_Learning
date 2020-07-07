#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 11:43:00 2020

@author: simon
"""


import torch
from ...HNN.models.MLP2H_Separable_Hamil_LF import MLP2H_Separable_Hamil_LF
from ...HNN.models.MLP2H_Separable_Hamil_VV import MLP2H_Separable_Hamil_VV
from ...HNN.models.MLP2H_Separable_Hamil_PV import MLP2H_Separable_Hamil_PV

class ML_integrator:
    '''list of class of ML integrator,
    since all of them use the same interface but just different pth'''
    
    def __init__(self, filename):
        '''base initializer for all ML 
        
        Parameters
        ----------
        filename : string
            filename is the name of the states that to be initialized
            
        Precaution
        ----------
        for states for LF, its must be written with LF code in the string so its recognizeable
        the same goes for VV and PV or other methods to be introduced 
        '''
        import os 
        uppath = lambda _path, n : os.sep.join(_path.split(os.sep)[:-n])
        base_dir = uppath(__file__, 2) # get the Integrator base_dir
        best_setting = torch.load(base_dir + '/states/{}.pth'.format(filename))
        if 'LF' in filename.upper() : #leapfrog
            self.ML_integrator = MLP2H_Separable_Hamil_LF(2,20) # this setting could be changed     
        elif 'VV' in filename.upper(): #velocity verlet
            self.ML_integrator = MLP2H_Separable_Hamil_VV(2,20) 
        elif 'PV' in filename.upper() : #position verlet
            self.ML_integrator = MLP2H_Separable_Hamil_PV(2,20)
            
        self.ML_integrator.set_n_stack(1) # set stack to 1
        self.ML_integrator.load_state_dict(best_setting[0]['state_dict'])

    def __call__(self, **state): #base integrator function to call the class as a function 
        '''this allows the ML integrator to be called'''
        device = 'cpu'
        q = torch.tensor(state['phase_space'].get_q(), dtype = torch.float32).squeeze().requires_grad_(True).to(device)
        p = torch.tensor(state['phase_space'].get_p(), dtype = torch.float32).squeeze().requires_grad_(True).to(device) 
    
        self.ML_integrator.eval()

        q_next, p_next = self.ML_integrator(q,p, state['time_step'])

        state['phase_space'].set_q(q_next.cpu().detach().numpy().reshape(-1,state['DIM']))
        state['phase_space'].set_p(p_next.cpu().detach().numpy().reshape(-1,state['DIM']))
        
        return state 

class position_verlet_ML(ML_integrator):
    def __init__(self, filename):
        ''' full documentation is written in the ML_integrator class'''
        super().__init__(filename)
        
class velocity_verlet_ML(ML_integrator):
    def __init__(self,filename):
        ''' full documentation is written in the ML_integrator class'''
        super().__init__(filename)
        
class leap_frog_ML(ML_integrator):
    def __init__(self, filename):
        ''' full documentation is written in the ML_integrator class'''
        super().__init__(filename)
        
