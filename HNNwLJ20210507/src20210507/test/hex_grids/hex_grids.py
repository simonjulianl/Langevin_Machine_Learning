import torch
import matplotlib.pyplot as plt

class hex_grids:

    ''' grids class to help make grids from each particle position'''

    _obj_count = 0

    def __init__(self, b = 0.7): # HK
        '''
        Parameters
        ----------
        b : grid interval      b = rij = 0.7 * sigma => potential ~ 254.99
        girds_18center   : hexagonal grids is 18 points at position (0,0) as center
        '''

        hex_grids._obj_count += 1
        assert(hex_grids._obj_count == 1), type(self).__name__ + ' has more than one object'

        self.b  = b

        gridb = torch.tensor([[-b *0.5, -b],[-b *0.5, b], [-b, 0], [b, 0], [b *0.5, -b], [b *0.5, b]])
        girdhb = torch.tensor([[0, 2*b], [0, -2*b], [-b *1.5, b],  [-b *1.5,-b], [b *1.5,b], [b *1.5,-b]])
        grid2b = 2*gridb

        self.grids_18center = torch.cat((grid2b,gridb,girdhb))
        # grids_18center.shape is [grids18, 2]

        self.grids_18center = self.grids_18center[torch.argsort(self.grids_18center[:, 0])] # sort along xaxis

        print('grids initialized : gridL ',b)

    def make_grids(self, phase_space):
        '''
        make_grids function to shift 18 grids points at (0,0) to each particle position as center

        :return
        shift grids to particle position as center
                shape is [nsamples, nparticle, grids18, DIM=(x,y)]
        '''

        q_list = phase_space.get_q()
        boxsize = phase_space.get_boxsize()

        q_list = torch.unsqueeze(q_list,dim=2)
        # q_list.shape is [nsamples, nparticle, 1, DIM=(x coord, y coord)]

        grids_shift = self.grids_18center + q_list
        # grids_shift.shape is [nsamples, nparticle, grids18, DIM=(x,y)]

        phase_space.adjust_real(grids_shift,boxsize)

        return  grids_shift


    def show_grids_nparticles(self, grids_list, q_list,boxsize):

        for i in range(1): # show two samples

            plt.title('sample {}'.format(i))
            plt.xlim(-boxsize/2, boxsize/2)
            plt.ylim(-boxsize / 2, boxsize / 2)
            plt.plot(grids_list[:,:,0], grids_list[:,:,1], marker='.', color='k', linestyle='none', markersize=12)
            plt.plot(q_list[:, :, 0], q_list[:, :, 1], marker='x', color='r', linestyle='none', markersize=12)
            plt.show()
            plt.close()