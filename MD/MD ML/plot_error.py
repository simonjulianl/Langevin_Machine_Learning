#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 20:37:16 2020

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

x_list = [i for i in np.arange(0.05,0.5 + 0.05, 0.05)]
integrator_list = [0.0, 0.00090, 0.00356, 0.00879, 0.01734, 0.02990, 0.04707, 0.06940, 0.09736, 0.13134]

ml_list = [0.0, 0.001869375, 0.002956, 0.0040615, 0.00468925, 0.0038515, 0.00607275, 0.00418375, 0.00457425, 0.004127]

fig = plt.figure(figsize = (10,10), constrained_layout=True)
plt.plot(x_list,integrator_list, label = "integration")
plt.xticks(np.arange(0.05,0.5 + 0.05, 0.05))
plt.yticks(np.arange(0., 0.15, 0.01))

plt.plot(x_list, ml_list , label = "ml")
fig.legend(loc = "upper left", bbox_to_anchor=(0.1,0.75))
plt.grid(True)
plt.show()

