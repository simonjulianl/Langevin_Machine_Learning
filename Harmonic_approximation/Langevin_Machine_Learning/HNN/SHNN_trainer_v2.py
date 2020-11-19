#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 18:14:34 2020

@author: simon
"""

import torch
import shutil 
from torch.utils.data import DataLoader
from .dataset import Hamiltonian_Dataset
import matplotlib.pyplot as plt 
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import copy 

class SHNN_trainer: 
    '''SHNN refers to Stacked Hamiltonian Neural Network trainer
    this is a trainer class to help train, validate, plot, and save '''
    
    def __init__(self, level, folder_name, **kwargs):
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
        self._curr_level = 1
        # number of training level, refering to how many stacked CNN to be trained
        
        try : # optimizer setting 
            self._optimizer = kwargs['optim']
            self._scheduler = kwargs.get('scheduler' , False)
            self._loss = kwargs['loss']
        except : 
            raise Exception('optimizer setting error, optim/loss not found ')
            
        try : #data loader and seed setting 
            self._batch_size = kwargs['batch_size'] # will be used for data loader setting 
            seed = kwargs.get('seed', 937162211) # default seed is 9 digit prime number

            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            
            shuffle = kwargs.get('shuffle', True) # default shuffle the dataloader
            num_workers = kwargs.get('num_wokers', 0)
            self._n_epochs = int(kwargs['epoch']) 
        
        except :
            raise Exception('epoch / batch_size not defined ')
            
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._current_epoch = 1
        
        DataLoader_Setting = {'num_workers' : num_workers, 'pin_memory': True, 'shuffle' : shuffle}
        
        try : #dataset setting
            Temperature = kwargs['Temperature_List']
            temperature_for_test=kwargs['temperature_for_test']
            sample = kwargs['sample'] # sample per temperature
            self._time_step = kwargs['time_step'] * kwargs['iterations'] 
            # this is big time step to be trained
            if kwargs['DIM'] != 2 :
                raise Exception('Not supported for Dimension is not 2')
        except : 
            raise Exception('Temperature_List for loading / sample not found ')
            
        self._train_dataset = Hamiltonian_Dataset(Temperature,
                                            sample,
                                            mode = 'train',
                                            **kwargs)
        
        self._validation_dataset = Hamiltonian_Dataset(Temperature,
                                                 sample, 
                                                 mode = 'validation',
                                                 **kwargs)
        
        self._test_dataset = Hamiltonian_Dataset(temperature_for_test,
                                                 sample, 
                                                 mode = 'test',
                                                 **kwargs)
        
        self._train_loader = DataLoader(self._train_dataset, 
                                        batch_size = self._batch_size,
                                        **DataLoader_Setting)
        
        self._validation_loader = DataLoader(self._validation_dataset,
                                             batch_size = self._batch_size,
                                             **DataLoader_Setting)
        
        self._test_loader = DataLoader(self._test_dataset,
                                             batch_size = self._batch_size,
                                             **DataLoader_Setting)
        
        self._base_test_loader = copy.deepcopy(self._test_loader)
        # used to check the performance for higher level, avoid shallow copying
        
        try : #architecture setting 
            self._model = kwargs['model'].double().to(self._device)
        except : 
            raise Exception('model not found')
            
        #initialize best model 
        self._best_validation_loss = float('inf') # arbitrary value 
        # to log all the data 
        self._writer = SummaryWriter('runs/{}'.format(folder_name)) # explicitly mention in runs folder


    def train_epoch(self):
        '''
        helper function to train each epoch

        Returns
        -------
        float
            train loss per epoch

        '''
        model = self._model.train() # fetch the model
        criterion = self._loss # fetch the loss
        
        train_loss = 0

        for batch_idx, (data,label) in enumerate(self._train_loader) : 
            # especially for HNN where we need in-graph gradient
            #print(data)  # 2xsamplesxNum_of_particlesxDIM
            #print(type(data)) # list
            #print('train_epoch')
            q_list = data[0].to(self._device).requires_grad_(True)
            p_list = data[1].to(self._device).requires_grad_(True)
            #print('q_list',q_list)
            #print('p_list',p_list)

            q_list_label = label[0].to(self._device)
            p_list_label = label[1].to(self._device)
            #print('q_list_label', q_list_label)
            #print('p_list_label', p_list_label)

            label = (q_list_label, p_list_label) # rebrand the label
            
            self._optimizer.zero_grad()
     
            try :
                prediction = model(q_list, p_list, self._time_step)
            except : 
                continue # this happens when the data batch length is 1
            loss = criterion(prediction, label)

            loss.backward()
            
            train_loss += loss.item() # get the scalar output
     
            self._optimizer.step()
            
            if self._scheduler :# if scheduler exists
                self._scheduler.step(loss)
                
        return train_loss / len(self._train_loader.dataset) # return the average
        
    def validate_epoch(self, validation_loader):
        '''
        helper function to validate each epoch

        Returns
        -------
        q_diff, p_diff , validation : tuples of float
            difference and validation loss per epoch

        '''
        model = self._model.eval() # fetch the model
        criterion = self._loss # fetch the loss
        
        validation_loss = 0
        
        #with torch.no_grad() should not be used as we need to differentiate intermediate variables

        for batch_idx, (data,label) in enumerate(validation_loader) :
            #cast to torch
            #print('batch_idx',batch_idx)
            q_list = data[0].to(self._device).requires_grad_(True)
            p_list = data[1].to(self._device).requires_grad_(True)
            #print('valid')
            #print('q_list',q_list)
            #print('p_list',p_list)

            q_list_label = label[0].to(self._device)
            p_list_label = label[1].to(self._device)
            #print('q_list_label', q_list_label)
            #print('p_list_label', p_list_label)

            label = (q_list_label, p_list_label)

            try :
                prediction = model(q_list, p_list, self._time_step)
            except :
                continue # when data length is 1

            loss = criterion(prediction, label)

            validation_loss += loss.item() # get the scalar output

        return validation_loss / len(validation_loader.dataset)  #return the average

    def record_best(self, validation_loss, filename = 'checkpoint.pth') :
        '''
        helper function to record the state after each training

        Parameters
        ----------
        validation_loss : float
            validation loss per epoch
        filename : string
            path to the saving the checkpoint
        '''
        
        is_best = validation_loss < self._best_validation_loss 
        self._best_validation_loss = min(validation_loss, self._best_validation_loss)
        
        state = ({
            'epoch' : self._current_epoch ,
            'state_dict' : self._model.state_dict(),
            'best_validation_loss' : self._best_validation_loss,
            'optimizer' : self._optimizer,
            'scheduler' : self._scheduler,
            'loss' : self._loss,
            'level' : self._curr_level,
            'batch_size' : self._batch_size,
            }, is_best ) 
   
        torch.save(state, filename)
        if is_best :
            # higher level is always prefered compared to lower level
            self._best_state = state[0]
            shutil.copyfile(filename, 'model_best.pth')
            
    def train_level(self, filename = 'checkpoint.pth'):
        '''
        Another helper function to train the whole network, plot the loss and 
        save the entire thing for each level of training. 
        
        Parameters
        ----------
        filename : string
            filename of the pth to be saved. Default is checkpoint.pth
            
        Precaution
        ----------
        Loss is saved in npy files with (train, validation) loss format
        '''
         
        if self._curr_level != 1 : 
            # if its not level 1 , check performance of previous level weight of more steps to compare
            validation_loss = self.validate_epoch(self._test_loader)
            print('performance of level {} weight on level {}'.format(self._curr_level - 1, self._curr_level))
            print('\t test loss : {:.6f}'.format(validation_loss))
            
        for i in range(1, self._n_epochs + 1):
            train_loss = self.train_epoch()
            validation_loss = self.validate_epoch(self._validation_loader)
            print('epoch:{} train_loss:{:.6f} ; validation_loss.{:.6f}'.format(i,train_loss,validation_loss))

            self.record_best(validation_loss, filename)
            self._current_epoch += 1
            
            self._writer.add_scalar('training loss_level {}'.format(self._curr_level),
                                    train_loss,
                                    i # epoch
                                    )
            
            self._writer.add_scalar('validation loss_level {}'.format(self._curr_level),
                        validation_loss,
                        i # epoch
                        )

        print('training level : {}'.format(self._curr_level))
        print('best setting : \n\t epoch : {} \n\t validation_loss : {:.6f}'.format(self._best_state['epoch'], 
                                                                                self._best_state['best_validation_loss']))

        
        #check the performace of current weight level on base level 
        self._model.load_state_dict(self._best_state['state_dict']) 
        #check performance on test loader 
        test_loss = self.validate_epoch(self._test_loader)
        
        print('performance on test dataset : \n\t test_loss : {:.5f}'.format(test_loss))
        #choose the best model from the previous level and pass it to the next level
        
        self._model.set_n_stack(1) # set the model level
        base_test_loss = self.validate_epoch(self._base_test_loader)
        
        print('performance on base level (1) : \n\t test_loss : {:.5f}'.format(base_test_loss))
          
    def up_level(self):
        '''helper function to shift the dataset and level'''
        
        self._curr_level += 1 # increase the level
        self._current_epoch = 1 # reset the number of current epoch 
        
        try:
            getattr(self._model, 'n_stack') 
        except :
            raise Exception('This model does not support stacking, self.n_stack not found')
            
        self._model.set_n_stack(self._curr_level) # set the model level
        
        self._train_dataset.shift_layer()
        self._validation_dataset.shift_layer() 
        self._test_dataset.shift_layer()
        # this is a function to label of the layer up for each dataset
        
        #change learning rate per level, tuning required 
        for param_group in self._optimizer.param_groups :
            param_group['lr'] = param_group['lr'] / 10
            #every level increase reduce lr by factor of 10
        
    def train(self):
        '''overall function to train the networks for different levels'''
        
        for i in range(self._level_epochs):
            print('current level : {}'.format(self._curr_level))
            self.train_level(filename = 'checkpoint_level{}.pth'.format(self._curr_level))

            self._best_validation_loss = float('inf') # reset the validation loss, always prefer higher levels

            if i + 1 != self._level_epochs :  
                # for last epoch do not need to up level
                self.up_level()
                      
        self._writer.close() # close writer to avoid collision
        
