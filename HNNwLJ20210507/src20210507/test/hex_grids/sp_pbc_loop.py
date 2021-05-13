from scipy.spatial import distance
import torch
import numpy as np
import time
import math

if __name__=='__main__':

    seed = 93711
    torch.manual_seed(seed)

    boxszie = 1
    npar=2
    ngrid=2
    DIM = 2
    xi       = torch.rand([npar,DIM])
    grid_pt  = torch.rand([ngrid,DIM])

    print('xi ',xi)

    xi_shift = []
    dpairs = []

    for dx in range(-1,2):
       for dy in range(-1,2):
           print('dx dy ',dx, ' ' , dy)
           xis = xi + torch.tensor([dx,dy])
           xi_shift.append(xis)
           dp = distance.cdist(xis,grid_pt)
           tdp = torch.from_numpy(dp)
           dpairs.append(tdp)

    print('dpairs ',dpairs)

    print('dp shape ',dpairs[0].shape)

    # dpairs[0].shape = [nparticle,grids]
    # dpairs.shape = [9, nparicle, grids]
   
    dpairs_torch = torch.stack(dpairs)

    dpairs_good = torch.min(dpairs_torch,dim=0)

    print('dpairs_t ',dpairs_torch)
    print('dpairs_g ',dpairs_good)

