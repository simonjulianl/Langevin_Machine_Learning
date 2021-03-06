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
        
        self._test_dataset = Hamiltonian_Dataset(Temperature,
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

    def get_figure_Hq(self, p = [0]): 
        from torch.autograd import grad
        '''helper function to plot hamiltonian for different p and q 
        
        Parameters
        ----------
        p : list of p for H(q)
        q : list of q for H(p)
        
        Precaution
        ----------
        q_list range : [-4,4)
        p_list range : [-6,6), this is for the sake of inference when T = 1, please adjust as you need
        
        Return
        ------
        fig : matplotlib.figure
            figure object created by matplotlib
            
        '''
        fig = plt.figure(figsize = (12, 6), dpi = 200)
        
        q_list = torch.tensor(np.arange(-4,4,0.01), # just arbitrary choice  
                              dtype = torch.float64).to(self._device).requires_grad_(True)
        
        idx = 1
        col_length = len(p)
        for p_const in p: 
            p_list = torch.ones(q_list.shape,
                                 dtype = torch.float64).to(self._device).requires_grad_(True) * p_const # keep constant at 0
            coordinates = torch.cat((q_list.unsqueeze(1), p_list.unsqueeze(1)), dim = 1)
            try :
                U_q = self._model.linear_potential(coordinates)
                dUdq = grad(U_q.sum(), q_list, create_graph = False)[0]
            except : 
                raise Exception('The model doesnot support linear_potential')
            
            x_value = list(q_list.cpu().detach().numpy())
            y_value = list(U_q.cpu().detach().numpy())
    
            ax = fig.add_subplot(2, col_length, idx)    
            ax.plot(x_value, y_value, color = 'orange', label = 'U(q)')
            ax.legend(loc = 'best')                        
            ax.grid(True)
            ax.set_xlabel('q /  position')
            ax.set_ylabel('Potential / H(q)')
            ax.set_title('plot of H(q) when p = {}'.format(p_const))
            
            #plot of derivative of U / H(q)
            y_value = list(dUdq.cpu().detach().numpy())
            ax = fig.add_subplot(2, col_length, idx + col_length)
            ax.plot(x_value, y_value, color = 'orange', label = 'dUdq')
            ax.legend(loc = 'best') 
            ax.grid(True)
            ax.set_ylabel('dUdq')
            ax.set_xlabel('q / position ')
            ax.set_title('plot of dH/dq, p = {}'.format(p_const))
            
            idx += 1 # increase index of plotting

        fig.tight_layout(pad = 2.0) # add spacing
        
        return fig 
    
    def get_figure_Hp(self, q = [0]): 
        from torch.autograd import grad
        '''helper function to plot hamiltonian for different p and q 
        
        Parameters
        ----------
        q : list of q for H(p)
        
        Precaution
        ----------
        q_list range : [-4,4)
        p_list range : [-6,6), this is for the sake of inference when T = 1, please adjust as you need
        
        Return
        ------
        fig : matplotlib.figure
            figure object created by matplotlib
            
        '''
        fig = plt.figure(figsize = (12, 6), dpi = 200)
        
        p_list = torch.tensor(np.arange(-6,6,0.01), # just arbitrary choice  
                              dtype = torch.float64).to(self._device).requires_grad_(True)
        
        idx = 1
        col_length = len(q)

        for q_const in q : 
            q_list = torch.ones(p_list.shape,
                                 dtype = torch.float64).to(self._device).requires_grad_(True) * q_const # keep constant at 0
            # H(p,q) = U(q) + KE(p) 
            coordinates = torch.cat((q_list.unsqueeze(1), p_list.unsqueeze(1)), dim = 1)
    
            try : 
                KE_p = self._model.linear_kinetic(coordinates)
                dKEdp = grad(KE_p.sum(), p_list, create_graph = False)[0]
            except : 
                raise Exception('The model doesnot support linear_kinetic')
            
            x_value = list(p_list.cpu().detach().numpy())
            y_value = list(KE_p.cpu().detach().numpy())
            
            ax = fig.add_subplot(2, col_length, idx)
            ax.plot(x_value, y_value, color ='orange', label = 'KE(p)')
            ax.grid(True)
            ax.legend(loc = 'best')
            ax.set_xlabel('p / momentum')
            ax.set_ylabel('KE / H(p)')
            ax.set_title('plot of H(p) when q = {}'.format(q_const))
            
            #plot derivative of KE / H(p)
            y_value = list(dKEdp.cpu().detach().numpy())
            ax = fig.add_subplot(2, col_length, idx + col_length)
            ax.plot(x_value, y_value , color = 'orange', label = 'dKEdp')
            ax.set_xlabel('p / momentum')
            ax.set_ylabel('dKEdp')
            ax.set_title('plot of dH/dp when q = {}'.format(q_const))
            ax.legend(loc = 'best') 
            ax.grid(True)
            idx += 1 #increase the indexing of plotting
        
        fig.tight_layout(pad = 2.0) # add spacing
        
        return fig 
    
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
            p_list_label = label[1][0].squeeze().to(self._device) 
     
            label = (q_list_label, p_list_label) # rebrand the label
 
            #for 1 dimensional data, squeeze is the same as linearize as N x 2 data
            
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
        q_diff, p_diff = 0, 0
        q_diff_2 , p_diff_2  = 0, 0 # to calculate variance and hence std
        for batch_idx, (data,label) in enumerate(validation_loader) : 
            #cast to torch 
            q_list = data[0][0].to(self._device).squeeze().requires_grad_(True)
            p_list = data[1][0].to(self._device).squeeze().requires_grad_(True)
            
            q_list_label = label[0][0].squeeze().to(self._device) 
            p_list_label = label[1][0].squeeze().to(self._device) 
            label = (q_list_label, p_list_label)
      
            try : 
                prediction = model(q_list, p_list, self._time_step)    
            except :
                continue # when data length is 1
                
            loss = criterion(prediction, label)
            q_diff += torch.sum(torch.abs(prediction[0] - label[0])).item()
            p_diff += torch.sum(torch.abs(prediction[1] - label[1])).item()
            
            q_diff_2 += torch.sum((prediction[0] - label[0]) ** 2.0).item()
            p_diff_2 += torch.sum((prediction[1] - label[1]) ** 2.0).item()
            validation_loss += loss.item() # get the scalar output
            
        q_diff /= len(validation_loader.dataset)
        p_diff /= len(validation_loader.dataset)
        
        q_diff_2 /= len(validation_loader.dataset)
        p_diff_2 /= len(validation_loader.dataset)
        
        return (q_diff, p_diff, validation_loss / len(validation_loader.dataset), q_diff_2, p_diff_2) #return the average 

    def record_best(self, validation_loss, q_diff, p_diff, q_diff_2, p_diff_2, filename = 'checkpoint.pth') : 
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
            'q_diff' : q_diff,
            'p_diff' : p_diff,
            'batch_size' : self._batch_size,
            'p_diff_2' : p_diff_2,
            'q_diff_2' : q_diff_2,
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
            q_diff, p_diff, validation_loss, q_diff_2, p_diff_2 = self.validate_epoch(self._test_loader)
            print('performance of level {} weight on level {}'.format(self._curr_level - 1, self._curr_level))
            std_q = (q_diff_2 - q_diff ** 2.0) ** 0.5
            std_p = (p_diff_2 - p_diff ** 2.0) ** 0.5
            
            print('\t q_diff : {:.5f} ± {:.5f} \t p_diff : {:.5f} ± {:.5f} \n\t test loss : {:.6f}'.format(
                q_diff, std_q, p_diff, std_p, validation_loss
                ))
            
        for i in range(1, self._n_epochs + 1):
            train_loss = self.train_epoch()
            q_diff, p_diff, validation_loss, q_diff_2, p_diff_2 = self.validate_epoch(self._validation_loader)
                     
            self.record_best(validation_loss, q_diff, p_diff, q_diff_2, p_diff_2, filename)
            self._current_epoch += 1
            
            self._writer.add_scalar('training loss_level {}'.format(self._curr_level),
                                    train_loss,
                                    i # epoch
                                    )
            
            self._writer.add_scalar('validation loss_level {}'.format(self._curr_level),
                        validation_loss,
                        i # epoch
                        )
            
            self._writer.add_scalar('q difference_level {}'.format(self._curr_level),
                                    q_diff,
                                    i # epoch
                                    )
            
            self._writer.add_scalar('p difference_level {}'.format(self._curr_level),
                        p_diff,
                        i # epoch
                        )
                    
        print('training level : {}'.format(self._curr_level))
        print('best setting : \n\t epoch : {} \n\t validation_loss : {:.6f}'.format(self._best_state['epoch'], 
                                                                                self._best_state['best_validation_loss']))
        
        std_q = (self._best_state['q_diff_2']- self._best_state['q_diff'] ** 2.0) ** 0.5
        std_p = (self._best_state['p_diff_2']- self._best_state['p_diff'] ** 2.0) ** 0.5
        print('\t q_diff : {} ± {} \t p_diff : {} ± {}'.format(self._best_state['q_diff'],
                                                                 std_q,
                                                                 self._best_state['p_diff'],
                                                                 std_p))
        
        #check the performace of current weight level on base level 
        self._model.load_state_dict(self._best_state['state_dict']) 
        #check performance on test loader 
        q_diff, p_diff, test_loss, q_diff_2, p_diff_2 = self.validate_epoch(self._test_loader)
        
        std_q = (q_diff_2 - q_diff ** 2.0) ** 0.5
        std_p = (p_diff_2 - p_diff ** 2.0) ** 0.5
        
        print('performance on test dataset : \n\t test_loss : {:.5f}'.format(test_loss))
        print('\t q_diff : {:.5f} ± {:.5f} \t p_diff : {:.5f} ± {:.5f} '.format(q_diff, std_q, p_diff, std_p))
        #choose the best model from the previous level and pass it to the next level
        
        self._model.set_n_stack(1) # set the model level
        q_diff, p_diff, base_test_loss, q_diff_2, p_diff_2 = self.validate_epoch(self._base_test_loader)
        std_q = (q_diff_2 - q_diff ** 2.0) ** 0.5
        std_p = (p_diff_2 - p_diff ** 2.0) ** 0.5
        
        print('performance on base level (1) : \n\t test_loss : {:.5f}'.format(base_test_loss))
        print('\t q_diff : {:.5f} ± {:.5f}\t p_diff : {:.5f} ± {:.5f}'.format(q_diff, std_q, p_diff, std_p ))
          
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
            
            #using the best state
            hp_figure = self.get_figure_Hp(q = [-2,-1,0,1,2]) # hamiltonian H(p)
            hq_figure = self.get_figure_Hq(p = [-4,-2,0,2,4]) # hamiltonian H(q)
            
            self._writer.add_figure('hamiltonian_q_level{}'.format(self._curr_level), 
                                    hq_figure,
                                    global_step = (i+1) * len(self._train_loader)) # number of training batch done
            
            self._writer.add_figure('hamiltonian_p_level{}'.format(self._curr_level), 
                                    hp_figure,
                                    global_step = (i+1) * len(self._train_loader)) # number of training batch done
            
            if i + 1 != self._level_epochs :  
                # for last epoch do not need to up level
                self.up_level()
                      
        self._writer.close() # close writer to avoid collision
        
