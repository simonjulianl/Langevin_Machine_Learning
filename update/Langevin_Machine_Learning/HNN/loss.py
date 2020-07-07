#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 10:58:30 2020

@author: simon
"""

import torch.nn as nn

def qp_MSE_loss(qp_quantities, label):
    '''
    custom loss to compute overall MSE Loss 

    Parameters
    ----------
    derivative_predicted : tuple
        tuple of length 2, with elements :
            -q_quantity : torch.tensor
            quantities related to q such as q itself and dqdt
            -p_quantity : torch.tensor 
            quantities related to p such as p itself and dpdt

    label : tuple of length 2 with elements :
        -q_quantity : torch.tensor
        label of related q quantities
        -p_quantity : torch.tensor 
        label of related p quantities

    
    Precaution
    -------
    Order Matter, hence please be careful of the order
    For this loss to work, q and p quantity loss is assumed to be symmetrical
    as each of them is a degree of freedom of its own and treated
    symmetrically using MSE in this case
    
    Returns
    -------
    loss : float
        Total loss calculated

    '''

    q_quantity, p_quantity = qp_quantities # unpack tuples

    q_label, p_label = label

    _reduction = 'sum' # to amplify the loss magnitude 
    criterion = nn.MSELoss(reduction = _reduction)
    loss = criterion(q_quantity, q_label) + criterion(p_quantity, p_label)
    return loss

