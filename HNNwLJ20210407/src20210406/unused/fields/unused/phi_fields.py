import torch
import matplotlib.pyplot as plt
import copy

class ljbox2gridimg:

    def __init__(self,lennard_jones,nsamples, npixels, DIM, maxcut=100):

        self._DIM = DIM
        self._lennard_jones = lennard_jones
        self._nsamples = nsamples
        self._npixels = npixels
        self._boxsize = self._lennard_jones.boxsize
        self._maxcut = maxcut * self._lennard_jones.get_epsilon()
        self._mincut = -8 * self._lennard_jones.get_epsilon() # actual minccut -6 and then give margin -2 = -8
        self._grid_list = self.build_gridpoint()


    def show_grid_nparticles(self, q_list, title):

        plt.title(title)
        plt.plot(self._grid_list[:,0], self._grid_list[:,1], marker='.', color='k', linestyle='none', markersize=12)
        plt.plot(q_list[:,:, 0], q_list[:,:, 1], marker='x', color='r', linestyle='none', markersize=12)
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


    def phi_field(self, phase_space, bc):

        self._phi_field = self._lennard_jones.phi_npixels(phase_space, bc, self._grid_list)
        self._phi_field = self._phi_field.reshape((-1, self._npixels, self._npixels))

        return self._phi_field


    def phi_max_min(self):

        # print('before mask', self._phi_field)

        for z in range(self._nsamples):

            mask = self._phi_field[z] > self._maxcut
            self._phi_field[z][mask] = self._maxcut # cannot do that !! keep the phi field. this just show image

        # print('after mask', self._phi_field)

        return self._phi_field


    def show_gridimg(self):

        norm_phi_field = (self._phi_field[0] - self._mincut) * 255 / (self._maxcut - self._mincut) # take one sample

        plt.imshow(norm_phi_field, cmap='gray') # one sample
        plt.colorbar()
        plt.clim(0, 255)
        plt.show()
        plt.close()
