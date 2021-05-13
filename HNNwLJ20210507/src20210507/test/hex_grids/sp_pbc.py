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

    dpairs = torch.zeros((9,npar,ngrid)) # 9 is no of boxes at center as xi_space
    shifts = torch.tensor([[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]])

    print(shifts)

    # xi.shape = [npar, DIM]
    # shifts.shape = [9,DIM]
    shifts = torch.unsqueeze(shifts,dim=1)
    # xi_shifts.shape = [9,npar,DIM]
    xi_shifts = xi + shifts

    print('xi_shift ',xi_shifts)

    for i in range (9):

        dp = distance.cdist(xi_shifts[i].numpy(), grid_pt.numpy())
        # ValueError: XA must be a 2-dimensional array.

        tdp = torch.from_numpy(dp)
        dpairs[i] = tdp

    # dpairs.shape = [9, nparicle, grids]
    print('dpairs_t ', dpairs)
    dpairs_good,_ = torch.min(dpairs, dim=0)
    # dpairs_good.shape is [nparicle, grids]
    print('dpairs_g ', dpairs_good)
