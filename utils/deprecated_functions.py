#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 17:13:15 2020

@author: simon
"""

#deprecated functions

@staticmethod
# def potential_energy( **configuration):
#     '''
#     Helper Function to get the energy of the state depending on the external potential energy
    
#     Parameters
#     ----------
#     potential : string
#         string representation of the potential U(q) where q is the position
#         for eg. (q**2 - 1) ** 2.0 + q 
        
#         the potential is assumed to be symmetrical around x, y and z axis
        
#     **configuration : configuration state consisting 
#         - N : int
#             total number of particles
#         - DIM : int
#             Dimension of the particles 
#         - BoxSize : float 
#             scaling of the box cell 
#         - potential : string
#             string expression of the potential energy in U(q) 
#             where Ux Uy and Uz are assumed to be symmetrical

#     Returns
#     -------
#     Average Potential Energy : float
#         Potential Energy must be induced due to external potential
#     '''
#     try : 
#         N = configuration['N']
#         DIM = configuration['DIM']
#         pos = configuration['pos']
#         potential = configuration['potential']
#     except : 
#         raise Exception('N / Dimension / Mass / Potential not supplied')
        
#     try : 
#         BoxSize = configuration['BoxSize']
#     except :
#         BoxSize = 1 
#         warnings.warn('BoxSize not supplied, set to 1')
        
#     ene_pot = 0.0 
#     for i in range(N):
#         real_pos = BoxSize * pos[i,:] # rescale each velocity according to the box size
#         for j in range(DIM):  # The potential is split into Ux , Uy, Uz
#             q = real_pos[j]
#             ene_pot += eval(potential) 

#     ene_pot_aver = 1.0 * ene_pot / N
    
#     return ene_pot_aver


# @staticmethod        
#     def plot_loss(loss : list, mode : str) :
#         '''
#         helper function to plot loss

#         Parameters
#         ----------
#         loss : list
#             np.array of loss 
#         mode : str
#             change the label for validation, train, test modes

#         Raises
#         ------
#         Exception
#             modes not found
#         '''
#         if mode not in ['validation', 'train', 'test']:
#             raise Exception('mode not found, please check the mode')
            
#         if mode == 'validation':
#             plt.plot(loss, color = 'blue', label='validation loss')
#         elif mode =='train' : 
#             plt.plot(loss, color = 'blue', label = 'train loss')
#         else : 
#             plt.plot(loss, color = 'blue', label = 'test loss')
            
#         plt.legend(loc = 'best')
#         plt.xlabel('epoch')
#         plt.show()       
# @staticmethod 
# def force(**configuration):
#     '''
#     Static function to get the force given the current state and potential

#     Parameters
#     ----------
#     **configuration : configuration state consisting
#         - N : int
#             total number of particles
#         - DIM : int
#             Dimension of the particles 
#         - BoxSize : float 
#             scaling of the box cell 
#         - force : string
#             string expression of the force dUdq
#             where Ux Uy and Uz are assumed to be symmetrical

#     Returns
#     -------
#     force : np.array (N X DIM shape)
#         force matrix based on external potential of current state

#     '''
#     try : 
#         N = configuration['N']
#         DIM = configuration['DIM']
#         pos = configuration['pos']
#         dUdq = configuration['force']
#     except : 
#         raise Exception('N / Dimension / Mass / Potential not supplied')
        
#     try : 
#         BoxSize = configuration['BoxSize']
#     except :
#         BoxSize = 1 
#         warnings.warn('BoxSize not supplied, set to 1')
        

#     force = np.zeros((N,DIM)) # placeholder for force
#     for i in range(N):
#         real_pos = BoxSize * pos[i,:] # rescale each velocity according to the box size
#         for j in range(DIM):  # The potential is split into Ux , Uy, Uz
#             q = real_pos[j] # q is used to eval force expression
#             force[i][j] = -eval(dUdq) # since F = -dU/dq

#     return force

#all plotting functions that are replaced by tensorboard

       #save the loss in train, validation format 
        # since we are using tensorboard, there is no need to explicitly save the data using numpy
        # np.save('loss_level{}.npy'.format(self._curr_level),
        #         np.array((train_loss_list, validation_loss_list)))


#plot loss, always see training curve
        
        # assert len(train_loss_list) == len(validation_loss_list)
        
        # plt.plot(train_loss_list, color = 'orange', label = 'train_loss')
        # plt.plot(validation_loss_list, color = 'blue', label = 'validation_loss')
        # plt.xlabel('epoch / level {}'.format(self._curr_level))
        # plt.ylabel('loss')
        # plt.legend(loc = 'best')
        # plt.show()