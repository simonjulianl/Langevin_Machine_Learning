#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
from ..Integrator.ML_linear_integrator import ML_linear_integrator
from .dataset import Hamiltonian_Dataset
from ..hamiltonian.pb import periodic_bc
from ..phase_space import phase_space
from .pair_wise_HNN import pair_wise_HNN
from torch.utils.data import DataLoader

class HNN_trainer:
    
    def __init__(self, **kwargs):

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
            torch.backends.cudnn.deterministic = True # Processing speed may be lower then when the models functions nondeterministically.
            torch.backends.cudnn.benchmark = False
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)

            shuffle = kwargs.get('shuffle', False) # default shuffle the data loader
            num_workers = kwargs.get('num_wokers', 0)

            self._n_epochs = int(kwargs['epoch'])
            self._sample = kwargs['N']
            self._init_config = kwargs['init_config']
            Temperature = kwargs['Temperature']
            kwargs['pb_q'] = periodic_bc()
            kwargs['phase_space'] = phase_space()

        except :
            raise Exception('epoch / batch_size not defined ')
            
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._current_epoch = 1
        self._setting = kwargs # save the setting

        DataLoader_Setting = {'num_workers' : num_workers, 'pin_memory': True, 'shuffle' : shuffle}

        try : #dataset setting
            if kwargs['DIM'] != 2 :
                raise Exception('Not supported for Dimension is not 2')
        except : 
            raise Exception('Temperature_List for loading / sample not found ')

        self._train_dataset = Hamiltonian_Dataset(Temperature,
                                                  self._sample,
                                                  mode='train',
                                                  **kwargs)

        self._train_loader = DataLoader(self._train_dataset,
                                        batch_size=self._batch_size,
                                        **DataLoader_Setting)

        try : #architecture setting 
            self._model = kwargs['models'].double().to(self._device)
        except : 
            raise Exception('models not found')

    def train_epoch(self):

        model = self._model.train() # fetch the models
        criterion = self._loss  # fetch the loss

        train_loss = 0

        for batch_idx, data in enumerate(self._train_loader):

            print('batch_idx : {}, batch size : {}'.format(batch_idx,self._batch_size))

            print('=== initial data ===')
            q_list = data[0][0].to(self._device).requires_grad_(True)
            p_list = data[0][1].to(self._device).requires_grad_(True)

            print('=== label data ===')
            q_list_label = data[1][0].to(self._device)
            p_list_label = data[1][1].to(self._device)
            print('==================')

            label = (q_list_label, p_list_label)

            _pair_wise_HNN = pair_wise_HNN(self._setting['hamiltonian'], q_list, p_list, self._model, **self._setting)
            self._setting['pair_wise_HNN'] = _pair_wise_HNN

            q_pred, p_pred = ML_linear_integrator(**self._setting).integrate(multicpu=False)
            q_pred = q_pred.reshape(-1, q_pred.shape[2], q_pred.shape[3])
            p_pred = p_pred.reshape(-1, p_pred.shape[2], p_pred.shape[3])

            pred = (q_pred, p_pred)

            loss = criterion(pred, label)

            self._optimizer.zero_grad() # defore the backward pass, use the optimizer object to zero all of the gradients for the variables
            loss.backward()  # backward pass : compute gradient of the loss wrt models parameters
            train_loss = loss.item()  # get the scalar output
            self._optimizer.step() # calling the step function on an optimizer makes an update to its parameters

            # if self._scheduler:  # if scheduler exists
            #     self._scheduler.step(loss)

        return train_loss

    def train(self):

        for i in range(1, self._n_epochs + 1):
            print('==================')
            print('epoch',i)
            print('==================')
            train_loss = self.train_epoch()

            print('epoch:{} train_loss:{:.6f}'.format(i,train_loss))

