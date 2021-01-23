import torch
import matplotlib.pyplot as plt

class dataset:

    def __init__(self, NoML_hamiltonian, linear_integrator, lennard_jones, epsilon, sigma, **state):

        self._NoML_hamiltonian = NoML_hamiltonian
        self._linear_itegrator = linear_integrator
        self._lennard_jones = lennard_jones
        self._epsilon = epsilon
        self._sigma = sigma
        self._state = state

    def max_min(self, potential_grid):

        minm = potential_grid[0].min()
        maxm = potential_grid[0].max()

        dij = 0.4  # q_ij = 0.3
        maxm_cut = 4 * self._epsilon * ((pow(self._sigma, 12) / pow(dij, 12)) - (pow(self._sigma, 6) / pow(dij, 6)))

        if maxm > maxm_cut:

            max_idx = torch.argmax(potential_grid[0])
            potential_grid[0][max_idx] = maxm_cut
            maxm = maxm_cut

        return minm, maxm

    def grid_list(self):

        self._grid_list = self._state['phase_space'].build_gridpoint(self._state['npixels'], self._state['boxsize'],
                                                               self._state['DIM'])
        print('grid list',self._grid_list)
        return self._grid_list

    def phi_field_initial(self, filename):

        q_list, p_list = self._state['phase_space'].read(filename, nsamples=self._state['nsamples_label'])
        print('load',q_list.dtype)
        print('grid',self.grid_list())

        self._state['phase_space'].set_q(q_list)
        self._state['phase_space'].set_grid(self.grid_list())

        self.phi_field_in = self._lennard_jones.phi_npixels(self._state['phase_space'], self._state['pb_q'])

        return self.phi_field_in

    def phi_field_next(self, filename):

        q_list, p_list = self._state['phase_space'].read(filename, nsamples=self._state['nsamples_label'])
        print('load',q_list)

        self._state['nsamples_cur'] = self._state['nsamples_ML']  # for one step
        self._state['tau_cur'] = self._state['tau_short']  # tau = 0.01
        self._state['MD_iterations'] = int(self._state['tau_short'] / self._state['tau_cur'])

        print('short time step {}'.format(self._state['tau_cur']))
        q_next_list, p_next_list = self._linear_itegrator(**self._state).integrate(self._NoML_hamiltonian)
        q_next_list = q_next_list.type(torch.float64)

        self._state['phase_space'].set_q(q_next_list)
        self._state['phase_space'].set_grid(self.grid_list())

        self.phi_field_nx = self._lennard_jones.phi_npixels(self._state['phase_space'], self._state['pb_q'])

        return self.phi_field_nx

    def show_img(self, phi_field):

        minm, maxm = self.max_min(phi_field)
        norm_potenital = (phi_field - minm) * 255 / (maxm - minm)
        norm_potenital = norm_potenital.reshape((self._state['npixels'], self._state['npixels']))
        plt.imshow(norm_potenital, cmap='gray')
        plt.colorbar()
        plt.clim(0, 255)
        plt.show()



