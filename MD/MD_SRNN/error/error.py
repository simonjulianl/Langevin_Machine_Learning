#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 10:58:35 2020

@author: simon
"""

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset,TensorDataset,DataLoader
from torchvision import datasets,transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm,trange

import os
import numpy as np

class Error(Exception):
    ''' base class '''
    pass

class InitializationError(Error):
    '''initialization error'''
    pass

class IntegrationError(Error):
    '''Integration Error'''
    pass

class SingletonError(Error):
    '''Singleton Error'''
    pass