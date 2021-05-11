import torch
dtype_long = torch.LongTensor

class grids:

    ''' grids class to help make grids and find coordinate, indices of grids'''

    _obj_count = 0

    def __init__(self, L, ng):
        '''
        Parameters
        ----------
        L : float     boxsize
        ng : int      gridL
        d  : grid interval
        '''

        grids._obj_count += 1
        assert(grids._obj_count == 1), type(self).__name__ + ' has more than one object'

        self.L   = L
        self.ng  = ng
        self.d   = L * 1.0 / ng
        self.L2d = L / (2*self.d)
        print('grids initialized : gridL ',ng,' boxsize ',L)

    def get_nearest_coord(self, q_list):
        '''
        :param q_list: shape is [ batch_size, nparticle, DIM ]

        :return:
        x0 : shape is [ batch_size, nparticle ]
        y0 : shape is [ batch_size, nparticle ]
        x0y0 : shape is [batch_size, nparticle, DIM ]
        '''

        i0batch = torch.floor(q_list[:,:,0] / self.d + self.L2d).type(dtype_long)
        j0batch = torch.floor(q_list[:,:,1] / self.d + self.L2d).type(dtype_long)

        i1batch = (i0batch + 1) % self.ng
        j1batch = (j0batch + 1) % self.ng

        i0j0 = torch.stack((i0batch, j0batch), dim=-1)
        i0j1 = torch.stack((i0batch, j1batch), dim=-1)
        i1j0 = torch.stack((i1batch, j0batch), dim=-1)
        i1j1 = torch.stack((i1batch, j1batch), dim=-1)

        x0  = -self.L / 2. + i0batch * self.d
        y0  = -self.L / 2. + j0batch * self.d
        x1 = x0 + self.d
        y1 = y0 + self.d

        x0y0 = torch.stack((x0, y0), dim=-1)
        x0y1 = torch.stack((x0, y1), dim=-1)
        x1y0 = torch.stack((x1, y0), dim=-1)
        x1y1 = torch.stack((x1, y1), dim=-1)

        return i0j0, i0j1, i1j0, i1j1, x0y0, x0y1, x1y0, x1y1

