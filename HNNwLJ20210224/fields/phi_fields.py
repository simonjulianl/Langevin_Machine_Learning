import torch
import matplotlib.pyplot as plt
from MC_parameters import MC_parameters
from MD_parameters import MD_parameters
import copy

class phi_fields:

    def __init__(self, npixels, hamiltonian, maxcut=100):

        terms = hamiltonian.get_terms()
        self.lennard_jones = terms[0]
        self._DIM = MC_parameters.DIM
        self._nsamples = MD_parameters.nsamples
        self._npixels = npixels
        self._boxsize = MC_parameters.boxsize
        self._maxcut = maxcut * MD_parameters.sigma
        self._mincut = -8 * MD_parameters.sigma # actual mincut -6 and then give margin -2 = -8 <= each grid can have nearest particles maximum 6
        self._grid_list = self.build_gridpoint()


    def show_grid_nparticles(self, q_list, title):

        #for i in range(MD_parameters.nsamples):
        for i in range(1):

            plt.title(title + 'sample {}'.format(i))
            plt.plot(self._grid_list[:,0], self._grid_list[:,1], marker='.', color='k', linestyle='none', markersize=12)
            plt.plot(q_list[i,:, 0], q_list[i,:, 1], marker='x', color='r', linestyle='none', markersize=12)
            plt.show()
            plt.close()

    def build_gridpoint(self):

        xvalues = torch.arange(0, self._npixels, dtype=torch.float64)
        xvalues = xvalues - 0.5 * self._npixels
        yvalues = torch.arange(0, self._npixels, dtype=torch.float64)
        yvalues = yvalues - 0.5 * self._npixels
        gridx, gridy = torch.meshgrid(xvalues * (self._boxsize / self._npixels) , yvalues * (self._boxsize / self._npixels) )
        grid_list = torch.stack([gridx, gridy], dim=-1)
        grid_list = grid_list.reshape((-1, self._DIM))

        return grid_list


    def phi_field(self, phase_space):

        self._phi_field = self.lennard_jones.phi_npixels(phase_space, self._grid_list)
        self._phi_field = self._phi_field.reshape((-1, self._npixels, self._npixels))

        # phi_max_min
        for z in range(0, len(self._phi_field), MD_parameters.nsamples_batch):

            mask = self._phi_field[z] > self._maxcut
            self._phi_field[z][mask] = self._maxcut

        return self._phi_field


    def show_gridimg(self, phi_field, time):

        #for i in range(MD_parameters.nsamples):
        for i in range(1):
            norm_phi_field = (phi_field[i] - self._mincut) * 255 / (self._maxcut - self._mincut) # take one sample

            plt.title(r'nparticle {}, npixels {}, boxsize {:.2f}, $\phi$-fields{}'.format(MC_parameters.nparticle, self._npixels, self._boxsize, time))
            plt.imshow(norm_phi_field, cmap='gray') # one sample
            plt.colorbar()
            plt.clim(0, 255)
            plt.show()
            plt.close()
