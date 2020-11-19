#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 18:14:34 2020

@author: simon
"""

import torch
import shutil 
from torch.utils.data import DataLoader
from ..utils.data_util import data_loader
from .dataset import Hamiltonian_Dataset
import matplotlib.pyplot as plt 
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
import copy 

class SHNN_tester:
    '''SHNN refers to Stacked Hamiltonian Neural Network trainer
    this is a trainer class to help train, validate, plot, and save '''
    
    def __init__(self, level, **kwargs):
        '''
        Initialize the class for the SHNN trainer
        SHNN is Stacked Hamiltonian Neural Network with modified implementation of HNN
        based on https://arxiv.org/abs/1906.01563. only for 1 dimensional data 

        Parameters
        ----------
        level : int
            number of levels of training, level one is (q,p) -> NN ->(q,p)
            level two is (q,p) -> NN ->(q,p) -> NN -> (q,p) and so on
            
        folder_name : string 
            name of tensorboard folder to be saved to log all the data 
            
        **kwargs : trainer_setting, containing : 
            
            optim : function
                torch.optim optimizer with full parameters and settings
            scheduler : function, optional
                torch.optim.lr_scheduler with full parameters and settings
            loss : function
                custom loss function or torch.nn losses 
            
            DataLoader and seed setting :
                
            epoch : int
                number of epoch
            seed : float, optional
                default is 937162211. for reproducibility
            batch_size : int
                batch size for batch update
            shuffle : boolean, optional
                shuffle data loader. Default is True
            num_workers : int, optional
                number of workers for dataloader. Default is 8
                
            Configuration setting :
                
            Temperature_List : list
                temperature for dataset
            sample : int 
                number of sample per temperature
            ** for complete configuration, see Hamiltonian Dataset setting 
            
            
            Architecture : 
            model : function
                model to be trained 
            
        Raises
        ------
        Exception
            Missing Parameters, check error message

        '''
        self._level_epochs = level
        # number of training level, refering to how many stacked CNN to be trained

        try : #data loader and seed setting 
            self._batch_size = kwargs['batch_size'] # will be used for data loader setting 
            seed = kwargs.get('seed', 937162211) # default seed is 9 digit prime number

            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            
            shuffle = kwargs.get('shuffle', False) # default shuffle the dataloader
            num_workers = kwargs.get('num_wokers', 0)
        
        except :
            raise Exception('epoch / batch_size not defined ')
            
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._current_epoch = 1
        
        self.DataLoader_Setting = {'num_workers' : num_workers, 'pin_memory': True, 'shuffle' : shuffle}
        
        try : #dataset setting
            self.temperature_for_test=kwargs['Temperature_for_test']
            self._sample = kwargs['sample'] # sample per temperature
            self._time_step = kwargs['time_step'] * kwargs['iterations']
            self._iterations = kwargs['iterations']
            self._gamma = kwargs['gamma']

            uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])
            base_dir = uppath(__file__, 2)
            init_path = base_dir + '/init/'
            self._particle = kwargs['particle']
            self._DIM = kwargs['DIM']

            # this is big time step to be trained
            if kwargs['DIM'] != 2 :
                raise Exception('Not supported for Dimension is not 2')
        except : 
            raise Exception('Temperature_List for loading / sample not found ')
            

        self._test_dataset =  data_loader.test_data(init_path,
                                             self.temperature_for_test,
                                             self._sample,
                                             self._particle) # wrapper for dataloader

        self._test_loader = DataLoader(self._test_dataset,
                                             batch_size = self._batch_size,
                                             **self.DataLoader_Setting)
        
        try : #architecture setting 
            self._model = kwargs['model'].double().to(self._device)
        except : 
            raise Exception('model not found')


    def trajectory(self, _test_loader):
        '''
        helper function to validate each epoch

        Returns
        -------
        q_diff, p_diff , validation : tuples of float
            difference and validation loss per epoch

        '''
        
        #with torch.no_grad() should not be used as we need to differentiate intermediate variables

        for batch_idx, data in enumerate(_test_loader) :
            #cast to torch
            #print('batch_idx',batch_idx)
            q_list = data[:,0].to(self._device).requires_grad_(True)
            p_list = data[:,1].to(self._device).requires_grad_(True)
            #print('test_loader batch')
            #print('q_list',q_list)
            #print('p_list',p_list)

            pred_q_, pred_p_ = self._model(q_list, p_list, self._time_step)
            #print('pred')
            #print('q',pred_q_)
            #print('p',pred_p_)
            # truncate according to samples
            if batch_idx == 0:  # first iteration, copy the data
                pred_q = pred_q_
                pred_p = pred_p_
            else:
                pred_q = torch.cat((pred_q, pred_q_))
                pred_p = torch.cat((pred_p, pred_p_))

            assert pred_q.shape == pred_p.shape  # shape of p and q must be the same

        #print('pred for test_loader')
        #print(pred_q,pred_p)
        return pred_q, pred_p   #return the average

    def test(self,filename):
        '''overall function to train the networks for different levels'''
        checkpoint = torch.load(filename) #torch.load() gives you a dictionary. That dictionary has not an eval function.
        try:
            checkpoint.eval()
        except AttributeError as error:
            print ('error')
        ### 'dict' object has no attribute 'eval'

        self._model.load_state_dict(checkpoint[0]['state_dict']) #upload the weights to your model.

        pred_q_list = np.zeros((int(self._iterations*self._time_step) + 1,self._sample,self._particle*self._DIM))
        pred_p_list = np.zeros((int(self._iterations*self._time_step) + 1,self._sample,self._particle*self._DIM))
        #print('pred_p_list',pred_p_list.shape)
        #print(self._test_dataset[:,0].shape)
        pred_q_list[0] = self._test_dataset[:,0]
        pred_p_list[0] = self._test_dataset[:,1]

        for i in range(1,int(self._iterations*self._time_step)+1):
            #print('current iteration :',i)
            if i == 1:
                pred_q, pred_p = self.trajectory(self._test_loader)

                # cpu -> go from a gpu Tensor to cpu Tensor;
                # detach -> call detach if the Tensor has associated gradients. When detach is needed,
                # you want to call detach before cpu. Otherwise, PyTorch will create the gradients associated
                # with the Tensor on the CPU then immediately destroy them when numpy is called.;
                # numpy -> go from a cpu Tensor to np.array
                pred_q_list[i] = pred_q.cpu().detach().numpy()
                pred_p_list[i] = pred_p.cpu().detach().numpy()

                #print('iteration :',i)
                #print("print for loop pred q p")
                #print('q',pred_q)
                #print('p',pred_p)

            else:
                #print('iteration :', i)

                self._test_loader = DataLoader(test_data,
                                               batch_size=self._batch_size,
                                               **self.DataLoader_Setting)

                pred_q, pred_p = self.trajectory(self._test_loader)
                #print("print for loop pred q p")
                #print('q', pred_q)
                #print('p', pred_p)
                pred_q_list[i] = pred_q.cpu().detach().numpy()
                pred_p_list[i] = pred_p.cpu().detach().numpy()

            next_q = torch.unsqueeze(pred_q.cpu(), dim=1)
            next_p = torch.unsqueeze(pred_p.cpu(), dim=1)  # N X 1 X  DIM
            test_data = torch.cat((next_q, next_p), dim=1)  # N X 2 XDIM

            #print('test_data',test_data)
            #print(test_data.shape)
            print("end interation {}".format(i))
        #print('ierations')
        #print('q',pred_q_list)
        #print('p',pred_p_list)
        phase_space = np.array((pred_q_list, pred_p_list))
        base_library = os.path.abspath('Langevin_Machine_Learning/init')
        filename = '/N{}_T{}_ts{}_iter10000_vv_gm{}_5000sampled_predicted.npy'.format(self._particle, self.temperature_for_test,self._time_step/self._iterations,self._gamma)
        file_path = base_library + filename
        np.save(file_path, phase_space)
