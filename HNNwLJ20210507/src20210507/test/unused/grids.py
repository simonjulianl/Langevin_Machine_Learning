import torch
dtype_long = torch.LongTensor

class interpolate:

    def __init__(self, L, ng):

        self.L   = L
        self.ng  = ng
        self.d   = L * 1.0 / ng
        self.L2d = L / (2*self.d)

    def grid_nearest_coord(self, q_list):
        '''
        :param q_list: shape is [ batch_size, nparticle, DIM ]

        :return:
        x0 : shape is [ batch_size, nparticle ]
        y0 : shape is [ batch_size, nparticle ]
        '''

        ibatch = torch.floor(q_list[:,:,0] / self.d + self.L2d).type(dtype_long)
        jbatch = torch.floor(q_list[:,:,1] / self.d + self.L2d).type(dtype_long)

        x0  = -self.L / 2. + ibatch * self.d
        y0  = -self.L / 2. + jbatch * self.d
        x1 = x0 + 1
        y1 = y0 + 1

        # x0y0 = torch.stack((x0, y0),dim=-1)
        # x0y1 = torch.stack((x0, y0 + 1),dim=-1)
        # x1y0 = torch.stack((x0 + 1, y0),dim=-1)
        # x1y1 = torch.stack((x0 + 1, y0 + 1), dim=-1)

        return x0, y0, x1, y1

    def grid_nearest_ij(self, q_list):
        '''
        :param q_list: shape is [ batch_size, nparticle, DIM ]

        :return:
        ibatch : shape is [ batch_size, nparticle]
        jbatch : shape is [ batch_size, nparticle]
        '''

        ibatch = torch.floor(q_list[:, :, 0] / self.d + self.L2d).type(dtype_long)
        jbatch = torch.floor(q_list[:, :, 1] / self.d + self.L2d).type(dtype_long)

        i1batch = ibatch + 1
        j1batch = jbatch + 1

        # i0j0   = torch.stack((ibatch, jbatch),dim=-1)
        # i0j1  = torch.stack((ibatch, jbatch + 1), dim=-1)
        # i1j0  = torch.stack((ibatch + 1, jbatch), dim=-1)
        # i1j1 = torch.stack((ibatch + 1, jbatch + 1), dim=-1)

        return ibatch, jbatch, i1batch, j1batch

    def find_nearest_phi_fields(self, im, ij):
        '''
        parameter
        ------------
        im  : shape is [ DIM, gridy, gridx]
        i0, j1  : shape is [nparticle]
        j0, j1  : shape is [nparticle]

        :return
        Ia  : shape is [nparticle,DIM]
        '''
        i0, i1, j0, j1 = ij
        print(im.shape)
        Ia = im[:,i0, j0]
        Ib = im[:,i0, j1]
        Ic = im[:,i1, j0]
        Id = im[:,i1, j1]
        # im[:, y0, x0].shape is [ DIM=(x,y), nparticle, nparticle ]
        # Ia.shape is [nparticle, DIM=(x,y)]
        print('I', Ia, Ib, Ic, Id)
        print(Ia.shape)
        return Ia, Ib, Ic, Id
