from scipy.spatial import distance
import torch

class dpair_pbc:

    _obj_count = 0

    def __init__(self):

        dpair_pbc._obj_count += 1
        assert (dpair_pbc._obj_count == 1),type(self).__name__ + " has more than one object"


    def xi_shifts(self, xi_state):
        '''
        xi_shift function to shift xi position 9 times that is no of box at center as xi_state

        Returns
        ----------
        xi_shift : torch.tensor
                shape is [9,npar,DIM]
        '''
        shifts = torch.tensor([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]])
        # shifts.shape = [9,DIM]

        shifts = torch.unsqueeze(shifts, dim=1)
        # shifts.shape = [9,1,DIM]

        return xi_state + shifts

    def cdist(self, xi_state, grids_list):
        '''
        Parameters
        ----------
        xi_space : torch.tensor
                dimensionless state
                shape is [nparticle, DIM]
        grids_list : torch.tensor
                dimensionless state
                shape is [nparticle*grids18, DIM=(x,y)]
        '''

        nparticle, DIM = xi_state.shape
        ngrids18, DIM = grids_list.shape

        xi_shifts = self.xi_shifts(xi_state)
        # xi_shifts.shape is [9,npar,DIM]
        dpairs = torch.zeros((9, nparticle, ngrids18))

        for i in range(9):

            dp = distance.cdist(xi_shifts[i].numpy(), grids_list.numpy(), 'euclidean')
            tdp = torch.from_numpy(dp)
            dpairs[i] = tdp

        dpairs_good, _ = torch.min(dpairs, dim=0)
        # dpairs_good.shape is [nparicle, grids]

        return dpairs_good

