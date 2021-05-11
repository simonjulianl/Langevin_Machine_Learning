import torch
import matplotlib.pyplot as plt

dtype_long = torch.LongTensor

class grids:

    ''' grids class to help make grids and find coordinate, indices of grids'''

    _obj_count = 0

    def __init__(self, ng): # HK
        '''
        Parameters
        ----------
        ng : int      gridL
        d  : grid interval
        '''

        grids._obj_count += 1
        assert(grids._obj_count == 1), type(self).__name__ + ' has more than one object'

        self.ng  = ng

        print('grids initialized : gridL ',ng)


    def show_grids_nparticles(self, grids_list, q_list):

        for i in range(2): # show two samples

            plt.title('sample {}'.format(i))
            plt.plot(grids_list[:,0], grids_list[:,1], marker='.', color='k', linestyle='none', markersize=12)
            plt.plot(q_list[i,:, 0], q_list[i,:, 1], marker='x', color='r', linestyle='none', markersize=12)
            plt.show()
            plt.close()

    def get_grid_list(self, phase_space):
        '''
        Parameters
        ----------
        grids_list : torch.tensor
                shape is [nsamples, gridL, gridL, DIM=(x coord, y coord)]
        '''

        nsamples, nparticle, DIM = phase_space.get_q().shape
        boxsize = phase_space.get_boxsize()

        xvalues = torch.arange(0, self.ng, dtype=torch.float64)
        xvalues = xvalues - 0.5 * self.ng
        yvalues = torch.arange(0, self.ng, dtype=torch.float64)
        yvalues = yvalues - 0.5 * self.ng

        # create grids list shape is [gridL * gridL, 2]
        gridx, gridy = torch.meshgrid(xvalues * (boxsize / self.ng) , yvalues * (boxsize / self.ng) )
        grids_list = torch.stack([gridx, gridy], dim=-1)
        grids_list = grids_list.reshape((-1, DIM))
        grids_list = torch.unsqueeze(grids_list, dim=0)

        ngrids_list = torch.repeat_interleave(grids_list, nsamples, dim=0)
        # ngrids_list.shape is [nsamples, gridL*gridL, DIM]

        return ngrids_list


    def get_nearest_coord(self, phase_space):  # HK
        '''
        :param q_list: shape is [ batch_size, nparticle, DIM ]

        :return:
        x0 : shape is [ batch_size, nparticle ]
        y0 : shape is [ batch_size, nparticle ]
        x0y0 : shape is [batch_size, nparticle, DIM ]
        '''

        q_list = phase_space.get_q()
        L = phase_space.get_boxsize()

        d   = L * 1.0 / self.ng
        L2d = L / (2 * d)

        i0batch = torch.floor(q_list[:,:,0] /  d +  L2d).type(dtype_long)
        j0batch = torch.floor(q_list[:,:,1] /  d +  L2d).type(dtype_long)

        i1batch = (i0batch + 1) % self.ng
        j1batch = (j0batch + 1) % self.ng

        i0j0 = torch.stack((i0batch, j0batch), dim=-1)
        i0j1 = torch.stack((i0batch, j1batch), dim=-1)
        i1j0 = torch.stack((i1batch, j0batch), dim=-1)
        i1j1 = torch.stack((i1batch, j1batch), dim=-1)

        x0  = -L / 2. + i0batch *  d
        y0  = -L / 2. + j0batch *  d
        x1 = x0 +  d
        y1 = y0 +  d

        x0y0 = torch.stack((x0, y0), dim=-1)
        x0y1 = torch.stack((x0, y1), dim=-1)
        x1y0 = torch.stack((x1, y0), dim=-1)
        x1y1 = torch.stack((x1, y1), dim=-1)

        return i0j0, i0j1, i1j0, i1j1, x0y0, x0y1, x1y0, x1y1

