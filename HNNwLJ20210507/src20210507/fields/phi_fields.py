import torch
import matplotlib.pyplot as plt

class phi_fields:  # SJ

    ''' phi_fields class to help calculate phi fields on grids '''

    _obj_count = 0

    def __init__(self, gridL, noML_hamiltonian, maxcut=20):
        '''
        Parameters
        ----------
        gridL : int
        hamiltonian : noML obj
        maxcut  : threshold for potential energy
        mincut  : -6 and then give margin -2 = -8 <= each grid can have nearest particles maximum 6
        '''

        phi_fields._obj_count += 1
        assert(phi_fields._obj_count == 1), type(self).__name__ + ' has more than one object'

        self.gridL = gridL
        terms = noML_hamiltonian.get_terms()
        self.lennard_jones = terms[0]
        self._maxcut = maxcut * self.lennard_jones.sigma
        self._mincut = -8 * self.lennard_jones.sigma

        print('phi_fields initialized : gridL ',gridL,' maxcut ',maxcut)

    def show_grids_nparticles(self, grids_list, q_list):

        for i in range(2): # show two samples

            plt.title('sample {}'.format(i))
            plt.plot(grids_list[:,0], grids_list[:,1], marker='.', color='k', linestyle='none', markersize=12)
            plt.plot(q_list[i,:, 0], q_list[i,:, 1], marker='x', color='r', linestyle='none', markersize=12)
            plt.show()
            plt.close()

    def build_grid_list(self, phase_space):
        '''
        Parameters
        ----------
        grids_list : torch.tensor
                shape is [gridL * gridL, DIM=(x coord, y coord)]
        '''

        nsamples, nparticle, DIM = phase_space.get_q().shape
        boxsize = phase_space.get_boxsize()

        xvalues = torch.arange(0, self.gridL, dtype=torch.float64)
        xvalues = xvalues - 0.5 * self.gridL
        yvalues = torch.arange(0, self.gridL, dtype=torch.float64)
        yvalues = yvalues - 0.5 * self.gridL

        # create grids list shape is [gridL * gridL, 2]
        gridx, gridy = torch.meshgrid(xvalues * (boxsize / self.gridL) , yvalues * (boxsize / self.gridL) )
        grids_list = torch.stack([gridx, gridy], dim=-1)
        grids_list = grids_list.reshape((-1, DIM))

        return grids_list


    def gen_phi_fields(self, phase_space):

        grids_list = self.build_grid_list(phase_space)
        # shape is [ gridL * gridL , DIM=(x,y) ]

        # self.show_grids_nparticles(grids_list, phase_space.get_q()'img1 ')

        self._phi_field = self.lennard_jones.phi_fields(phase_space, grids_list)
        # shape is [ nsamples, gridL * gridL ]

        self._phi_field = torch.log(self._phi_field + 8 )

        self._phi_field = self._phi_field.reshape((-1, self.gridL, self.gridL))
        # shape is [ nsamples, gridL, gridL ]

        # phi_max_min
        mask = self._phi_field > self._maxcut
        self._phi_field[mask] = self._maxcut

        return self._phi_field


    def show_gridimg(self, phi_field, time):

        for i in range(2):  # show two samples

            norm_phi_field = (phi_field[i] - self._mincut) * 255 / (self._maxcut - self._mincut) # take one sample

            plt.title(r'gridL{}, $\phi$-fields{}'.format(self.gridL , time))
            plt.imshow(norm_phi_field, cmap='gray') # one sample
            plt.colorbar()
            plt.clim(0, 255)
            plt.show()
            plt.close()
