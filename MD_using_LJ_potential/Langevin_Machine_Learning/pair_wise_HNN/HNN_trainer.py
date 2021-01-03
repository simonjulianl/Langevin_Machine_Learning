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
            torch.backends.cudnn.deterministic = True # Processing speed may be lower then when the model functions nondeterministically.
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
            self._model = kwargs['model'].double().to(self._device)
        except : 
            raise Exception('model not found')

        self.q_label, self.p_label = self._train_dataset.data_label()

    def _MLdHdq(self):

        model = self._model.train() # fetch the model
        MLdHdq = torch.empty(self._setting['particle'],self._setting['DIM'])
        print(MLdHdq.shape)

        for batch_idx, data in enumerate(self._train_loader):

            print('batch_idx',batch_idx)
            print(data)     # nsamples x N_particle x (N_particle-1) x  (del_qx, del_qy, del_px, del_py, tau )
            print(data.shape)

            self._optimizer.zero_grad()

            MLdHdq[batch_idx] = model(data)

            print('MLdHdq{}'.format(batch_idx),MLdHdq[batch_idx])

        print('MLdHdq',MLdHdq)
        print('MLdHdq', MLdHdq.shape)
        return MLdHdq  # return the average

    def train_epoch(self):

        criterion = self._loss  # fetch the loss

        label = (self.q_label, self.p_label)  # rebrand the label

        _dHdq = pair_wise_HNN(self._setting['hamiltonian'], self._MLdHdq())
        self._setting['pair_wise_HNN'] = _dHdq
        print(self._setting)

        q_data, p_data = ML_linear_integrator(**self._setting).integrate(multicpu=False)
        q_data = q_data.reshape(-1,q_data.shape[2],q_data.shape[3])
        p_data = p_data.reshape(-1, p_data.shape[2], p_data.shape[3])

        data = (q_data,p_data)

        loss = criterion(torch.Tensor(data), torch.Tensor(label))

        loss.backward()

        train_loss = loss.item()  # get the scalar output

        self._optimizer.step()

        if self._scheduler:  # if scheduler exists
            self._scheduler.step(loss)

        return train_loss  # return the average

    def train(self):

        for i in range(1, self._n_epochs + 1):
            print('epoch',i)
            train_loss = self.train_epoch()
            print('epoch:{} train_loss:{:.6f} ; validation_loss.{:.6f}'.format(i,train_loss,train_loss))

