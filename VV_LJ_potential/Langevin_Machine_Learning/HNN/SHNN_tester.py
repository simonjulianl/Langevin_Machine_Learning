#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 18:14:34 2020

@author: simon
"""
import gc
import torch
import shutil 
from torch.utils.data import DataLoader
from ..utils.data_util import data_loader
from .dataset import Hamiltonian_Dataset
import matplotlib.pyplot as plt 
import numpy as np
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import copy 

class SHNN_tester:
    '''SHNN refers to Stacked Hamiltonian Neural Network trainer
    this is a trainer class to help train, validate, plot, and save '''
    
    def __init__(self, level, **kwargs):

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
            
            shuffle = kwargs.get('shuffle', False) # for test
            num_workers = kwargs.get('num_wokers', 0)
        
        except :
            raise Exception('epoch / batch_size not defined ')
            
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._current_epoch = 1
        
        self.DataLoader_Setting = {'num_workers' : num_workers, 'pin_memory': True, 'shuffle' : shuffle}
        
        try : #dataset setting
            self.temperature_for_test=kwargs['Temperature_for_test']
            self._sample = kwargs['sample'] # sample per temperature
            self._time_step = kwargs['time_step']
            self._iterations = kwargs['iterations']
            self._gamma = kwargs['gamma']
            self._retrain_num = kwargs['retrain_num']

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
        #now = datetime.now()
        #torch.cuda.empty_cache()
        for batch_idx, data in enumerate(_test_loader) :
            #cast to torch

            q_list = data[:,0].to(self._device).requires_grad_(True)
            p_list = data[:,1].to(self._device).requires_grad_(True)

            pred_q_, pred_p_ = self._model(q_list, p_list, self._time_step)

            # truncate according to samples
            if batch_idx == 0:  # first iteration, copy the data
                pred_q = pred_q_.cpu()
                pred_p = pred_p_.cpu()

                del pred_q_
                del pred_p_
                #gc.collect()
                #torch.cuda.empty_cache()
                #print(torch.cuda.max_memory_allocated(),torch.cuda.max_memory_cached())

                #delete tensor right away,so that we indeed only track the total memory used.

            else:
                pred_q = torch.cat((pred_q, pred_q_.cpu()))
                pred_p = torch.cat((pred_p, pred_p_.cpu()))

                del pred_q_
                del pred_p_


            assert pred_q.shape == pred_p.shape  # shape of p and q must be the same

        return pred_q, pred_p   #return the average


    def test(self,filename):
        '''overall function to train the networks for different levels'''
        checkpoint = torch.load(filename) #torch.load() gives you a dictionary. That dictionary has not an eval function.
        print('epochs that is best model:',checkpoint[0]['epoch'])
        print('valid loss that is best model:', checkpoint[0]['best_validation_loss'])
        try:
            checkpoint.eval()
        except AttributeError as error:
            print ('load a dictionary')
        ### 'dict' object has no attribute 'eval'

        self._model.load_state_dict(checkpoint[0]['state_dict']) #upload the weights to your model.

        pred_q_list = np.zeros((self._iterations + 1,self._sample,self._particle*self._DIM))
        pred_p_list = np.zeros((self._iterations + 1,self._sample,self._particle*self._DIM))

        pred_q_list[0] = self._test_dataset[:,0]
        pred_p_list[0] = self._test_dataset[:,1]

        for i in range(1,self._iterations+1):

            if i == 1:
                pred_q, pred_p = self.trajectory(self._test_loader)
                # cpu -> go from a gpu Tensor to cpu Tensor;
                # detach -> tell pytorch that you do not want to compute gradients for that variable
                # numpy -> go from a cpu Tensor to np.array
                pred_q_list[i] = pred_q.detach().numpy()
                pred_p_list[i] = pred_p.detach().numpy()

            else:

                self._test_loader = DataLoader(test_data,
                                               batch_size=self._batch_size,
                                               **self.DataLoader_Setting)

                pred_q, pred_p = self.trajectory(self._test_loader)
                pred_q_list[i] = pred_q.detach().numpy()
                pred_p_list[i] = pred_p.detach().numpy()

            next_q = torch.unsqueeze(pred_q, dim=1)
            next_p = torch.unsqueeze(pred_p, dim=1)  # N X 1 X  DIM
            test_data = torch.cat((next_q, next_p), dim=1)  # N X 2 XDIM

            print("end interation {}".format(i))

        phase_space = np.array((pred_q_list, pred_p_list))
        base_library = os.path.abspath('Langevin_Machine_Learning/init')
        filename_ = '/N{}_T{}_ts{}_iter{}_vv_gm{}_{}sampled_predicted_{}.npy'.format(self._particle, self.temperature_for_test,self._time_step,self._iterations,self._gamma,self._sample,self._retrain_num)
        file_path = base_library + filename_
        np.save(file_path, phase_space)
