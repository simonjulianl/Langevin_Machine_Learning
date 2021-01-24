import torch
import matplotlib.pyplot as plt

class ljbox2gridimg:

    def __init__(self, NoML_hamiltonian, lennard_jones, linear_integrator, filename, **state):

        self._NoML_hamiltonian = NoML_hamiltonian
        self._lennard_jones = lennard_jones
        self._linear_itegrator = linear_integrator

        self._state = state
        self._q_list_in = self.q_list_in(filename)
        self._q_list_nx = self.q_list_next()
        self._grid_list = self.grid_list()
        self._epsilon = state['epsilon']
        self._sigma = state['sigma']


    def grid_list(self):

        _grid_list = self._state['phase_space'].build_gridpoint(self._state['npixels'], self._state['boxsize'],
                                                               self._state['DIM'])

        return _grid_list

    def q_list_in(self, filename):

        _q_list_in, _ = self._state['phase_space'].read(filename, nsamples=self._state['nsamples_label'])

        return _q_list_in

    def q_list_next(self):

        self._state['phase_space'].set_q(self._q_list_in)

        self._state['nsamples_cur'] = self._state['nsamples_label']  # for one step
        self._state['tau_cur'] = self._state['tau_short']  # tau = 0.01
        self._state['MD_iterations'] = int(self._state['tau_short'] / self._state['tau_cur'])  # for one step

        print('short time step {}, iteration {}'.format(self._state['tau_cur'], self._state['MD_iterations']))
        _q_list_nx, _ = self._linear_itegrator(**self._state).integrate(self._NoML_hamiltonian)
        _q_list_nx = _q_list_nx.type(torch.float64)

        return _q_list_nx

    def show_grid_nparticles(self, q_list, title):

        plt.title(title)
        plt.plot(self._grid_list[:,0], self._grid_list[:,1], marker='.', color='k', linestyle='none', markersize=12)
        plt.plot(q_list[:,:, 0], q_list[:,:, 1], marker='x', color='r', linestyle='none', markersize=12)
        plt.show()
        plt.close()

    def phi_max_min(self, phi_field):

        minm = - self._state['nparticle']
        # minm = torch.max(phi_field[0])  # phi_field shape : nsamples x npixels, here take one sample that is index=0
        maxm = torch.max(phi_field[0])
        print('min, max',minm, maxm)

        dij = 0.1 # q_ij = 0.1
        maxm_cut = 4 * self._epsilon * ((pow(self._sigma, 12) / pow(dij, 12)) - (pow(self._sigma, 6) / pow(dij, 6)))

        if maxm > maxm_cut:
            print('phi maxm too big so that cut maxm={:.4f} at dij={}'.format(maxm_cut, dij))
            # max_idx = torch.argmax(phi_field[0])
            # phi_field[0][max_idx] = maxm_cut  # cannot do that !! keep the phi field. this just show image
            maxm = maxm_cut

        return minm, maxm

    def phi_field_initial(self):

        self._state['phase_space'].set_q(self._q_list_in)
        self._state['phase_space'].set_grid(self._grid_list)
        self._phi_field_in = self._lennard_jones.phi_npixels(self._state['phase_space'], self._state['pb_q'])
        self._phi_field_in = self._phi_field_in.reshape((-1, self._state['npixels'], self._state['npixels']))

        return self._phi_field_in

    def phi_field_next(self):

        self._state['phase_space'].set_q(self._q_list_nx)
        self._state['phase_space'].set_grid(self._grid_list)

        self._phi_field_nx = self._lennard_jones.phi_npixels(self._state['phase_space'], self._state['pb_q'])
        self._phi_field_nx = self._phi_field_nx.reshape((-1, self._state['npixels'], self._state['npixels']))

        return self._phi_field_nx

    def show_gridimg(self, phi_field):

        minm, maxm = self.phi_max_min(phi_field)
        norm_phi_field = (phi_field - minm) * 255 / (maxm - minm)

        # norm_phi_field = norm_phi_field.reshape((self._state['npixels'], self._state['npixels']))
        plt.imshow(norm_phi_field[0], cmap='gray') # one sample
        plt.colorbar()
        plt.clim(0, 255)
        plt.show()
        plt.close()
