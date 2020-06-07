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

class SHNN_trainer: 
    '''SHNN refers to Stacked Hamiltonian Neural Network trainer
    this is a trainer class to help train, validate, plot, and save '''
    
    def __init__(self, level = 1, **kwargs):
        '''
        Initialize the class for the SHNN trainer
        SHNN is Stacked Hamiltonian Neural Network with modified implementation of HNN
        based on https://arxiv.org/abs/1906.01563. only for 1 dimensional data 

        Parameters
        ----------
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
            num_workers = kwargs.get('num_wokers', 8)
            self._n_epochs = int(kwargs['epoch']) 
        
        except :
            raise Exception('epoch / batch_size not defined ')
            
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._current_epoch = 1
        
        DataLoader_Setting = {'num_workers' : num_workers, 'pin_memory': True, 'shuffle' : shuffle}
        
        try : #dataset setting
            Temperature = kwargs['Temperature_List']
            sample = kwargs['sample'] # sample per temperature
            self._time_step = kwargs['time_step'] * kwargs['iterations'] 
            # this is big time step to be trained
            if kwargs['DIM'] != 1 : 
                raise Exception('Not supported for Dimension is not 1')
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
        
        self._train_loader = DataLoader(self._train_dataset, 
                                        batch_size = self._batch_size,
                                        **DataLoader_Setting)
        
        self._validation_loader = DataLoader(self._validation_dataset,
                                             batch_size = self._batch_size,
                                             **DataLoader_Setting)
        
        try : #architecture setting 
            self._model = kwargs['model'].to(self._device)
        except : 
            raise Exception('model not found')
            
        #initialize best model 
        self._best_validation_loss = float('inf') # arbitrary value 
       
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
            q_list = data[0][0].to(self._device).squeeze().requires_grad_(True)
            p_list = data[1][0].to(self._device).squeeze().requires_grad_(True)

            q_list_label = label[0][0].squeeze().to(self._device) 
            p_list_label = label[0][0].squeeze().to(self._device) 
            label = (q_list_label, p_list_label) # rebrand the label
            #for 1 dimensional data, squeeze is the same as linearize as N x 2 data
            
            self._optimizer.zero_grad()
            
            prediction = model(q_list, p_list, self._time_step)

            loss = criterion(prediction, label)
            loss.backward()
            
            train_loss += loss.item() # get the scalar output
            
            self._optimizer.step()
            
            if self._scheduler :# if scheduler exists
                self._scheduler.step(loss)
                
        return train_loss / len(self._train_loader.dataset) # return the average
        
    def validate_epoch(self):
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
        q_diff, p_diff = 0, 0
        for batch_idx, (data,label) in enumerate(self._validation_loader) : 
            #cast to torch 
            q_list = data[0][0].to(self._device).squeeze().requires_grad_(True)
            p_list = data[1][0].to(self._device).squeeze().requires_grad_(True)
            
            q_list_label = label[0][0].squeeze().to(self._device) 
            p_list_label = label[0][0].squeeze().to(self._device) 
            label = (q_list_label, p_list_label)
            
            prediction = model(q_list, p_list, self._time_step) 
            loss = criterion(prediction, label)
            q_diff += torch.sum(torch.abs(prediction[0] - label[0])).item()
            p_diff += torch.sum(torch.abs(prediction[1] - label[1])).item()
            validation_loss += loss.item() # get the scalar output
                
        q_diff /= len(self._validation_loader.dataset)
        p_diff /= len(self._validation_loader.dataset)
        
        return (q_diff, p_diff, validation_loss / len(self._validation_loader.dataset) ) #return the average 
    
    def record_best(self, train_loss, validation_loss, q_diff, p_diff, filename = 'checkpoint.pth') : 
        '''
        helper function to record the state after each training

        Parameters
        ----------
        train_loss : float
            training loss per epoch
        validation_loss : float
            validation loss per epoch
        filename : string
            path to the saving the checkpoint
        '''
        #output the loss
        print('Epoch : {} \n\t Average Train Loss : {:.6f} \n\t Average Validation Loss : {:.6f}'.format(
                self._current_epoch, train_loss, validation_loss
                ))
        print('\t q diff : {:5f} \t p diff : {:5f}'.format(q_diff, p_diff))
        
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
            'q_diff' : q_diff,
            'p_diff' : p_diff,
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
         
        train_loss_list = []
        validation_loss_list = []
        for i in range(1, self._n_epochs + 1):
            train_loss = self.train_epoch()
            q_diff, p_diff, validation_loss = self.validate_epoch()
            
            self.record_best(train_loss, validation_loss, q_diff, p_diff, filename)
            self._current_epoch += 1
            
            train_loss_list.append(train_loss)
            validation_loss_list.append(validation_loss)
            
        print('training level : {}'.format(self._curr_level))
        print('best setting : \n\t epoch : {} \n\t validation_loss : {:.6f}'.format(self._best_state['epoch'], 
                                                                                self._best_state['best_validation_loss']))
        print('\t q_diff : {} \t p_diff : {}'.format(self._best_state['q_diff'],
                                                  self._best_state['p_diff']))
        #plot loss, always see training curve
        assert len(train_loss_list) == len(validation_loss_list)
        
        plt.plot(train_loss_list, color = 'orange', label = 'train_loss')
        plt.plot(validation_loss_list, color = 'blue', label = 'validation_loss')
        plt.xlabel('epoch / level {}'.format(self._curr_level))
        plt.ylabel('loss')
        plt.legend(loc = 'best')
        plt.show()
        
        #save the loss in train, validation format 
        np.save('loss_level{}.npy'.format(self._curr_level),
                np.array((train_loss_list, validation_loss_list)))
        
        del train_loss_list, validation_loss_list 
        
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
        # this is a function to label of the layer up for each dataset
        
    def train(self):
        '''overall function to train the networks for different levels'''
        
        for i in range(self._level_epochs):
            self.train_level(filename = 'checkpoint_level{}.pth'.format(self._curr_level))
            # for last epoch do not need to up level
            if i + 1 != self._level_epochs :  
                self._best_validation_loss = float('inf') # reset the validation loss
                #choose the best model from the previous level and pass it to the next level
                self._model.load_state_dict(self._best_state['state_dict'])
                self.up_level()
        