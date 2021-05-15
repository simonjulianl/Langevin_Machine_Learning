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
    ngrid=4
    DIM = 2
    xi       = torch.rand([npar,DIM])
    grid_pt  = torch.rand([ngrid,DIM])

    # print('xi ',xi)

    dpairs = torch.zeros((9,npar,ngrid)) # 9 is no of boxes at center as xi_space
    shifts = torch.tensor([[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]])
    # shifts.shape = [9,DIM]
    # print(shifts)

    # xi.shape = [npar, DIM]
    shifts = torch.unsqueeze(shifts,dim=1)
    # shifts.shape = [9,1,DIM]

    xi_shifts = xi + shifts
    # xi_shifts.shape = [9,npar,DIM]

    # print('xi_shift ',xi_shifts)

    for i in range (9):

        # dp = distance.cdist(xi_shifts[i].numpy(), grid_pt.numpy())
        # # ValueError: XA must be a 2-dimensional array.
        # # grid_pt.shape is [ngrids, DIM]
        # tdp = torch.from_numpy(dp)
        # dpairs[i] = tdp

        dp = torch.cdist(xi_shifts[i], grid_pt)
        dpairs[i] = dp

    # dpairs.shape = [9, nparicle, grids]
    print('dpairs_t ', dpairs)
    dpairs_good,_ = torch.min(dpairs, dim=0)
    # dpairs_good.shape is [nparicle, grids]
    print('dpairs_g ', dpairs_good)

