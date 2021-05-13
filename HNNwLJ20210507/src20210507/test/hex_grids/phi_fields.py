import torch
from hex_grids import hex_grids
import matplotlib.pyplot as plt

class phi_fields:  # HK

    ''' phi_fields class to help calculate phi fields on grids '''

    _obj_count = 0

    def __init__(self, noML_hamiltonian, grids18=18, maxcut=20): # HK
        '''
        Parameters
        ----------
        grids18 : int
        hamiltonian : noML obj
        maxcut  : threshold for potential energy
        mincut  : -6 and then give margin -2 = -8 <= each grid can have nearest particles maximum 6
        '''

        phi_fields._obj_count += 1
        assert(phi_fields._obj_count == 1), type(self).__name__ + ' has more than one object'

        self.grids18 = grids18
        terms = noML_hamiltonian.get_terms()
        self.lennard_jones = terms[0]
        self._maxcut = maxcut * self.lennard_jones.sigma
        self._mincut = -8 * self.lennard_jones.sigma

        self.hex_grids = hex_grids()
        print('phi_fields initialized : grids ',grids18,' maxcut ',maxcut)


    def gen_phi_fields(self, phase_space):

        _, nparticle, DIM = phase_space.get_q().shape

        grids_list = self.hex_grids.make_grids(phase_space)
        # shape is [nsamples, nparticle, grids18, DIM=(x,y)]

        self.hex_grids.show_grids_nparticles(grids_list[0], phase_space.get_q(), phase_space.get_boxsize())  # show about one sample

        grids_list = grids_list.reshape(-1,grids_list.shape[1]*grids_list.shape[2],grids_list.shape[3])
        # shape is [nsamples, nparticle*grids18, DIM=(x,y)]

        self._phi_field = self.lennard_jones.phi_fields(phase_space, grids_list)
        # shape is [ nsamples, nparticle*grids18 ]

        self._phi_field = torch.log(self._phi_field + 8 ) # HK why +8??
        # shape is [ nsamples, nparticle*grids18 ]

        self._phi_field = self._phi_field.reshape((-1, nparticle, self.grids18))
        # shape is [ nsamples, nparticle, grids18 ]

        # phi_max_min
        mask = self._phi_field > self._maxcut
        self._phi_field[mask] = self._maxcut

        return self._phi_field

    def show_gridimg(self, phi_field, time):

        for i in range(1):  # show two samples

            norm_phi_field = (phi_field[i] - self._mincut) * 255 / (self._maxcut - self._mincut) # take one sample

            plt.title(r'gridL{}, $\phi$-fields{}'.format(self.grids18 , time))
            plt.imshow(norm_phi_field, cmap='gray') # one sample
            plt.colorbar()
            plt.clim(0, 255)
            plt.show()
            plt.close()
