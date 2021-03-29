import torch
import matplotlib.pyplot as plt
import copy

class ljbox2gridimg:

    # def __init__(self, NoML_hamiltonian, lennard_jones, linear_integrator, filename, **state):

    def __init__(self,lennard_jones, npixels=64, maxcut=100):
        # self._NoML_hamiltonian = NoML_hamiltonian
        self._lennard_jones = lennard_jones
        self._npixels = npixels
        self._maxcut = maxcut
        self._mincut = -8*self._lennard_jones.get_epsilon()
        # self._linear_itegrator = linear_integrator

        # self._state_local_copy = copy.deepcopy(state)
        # self._q_list_in, self._p_list_in = self.qnp_list_in(filename)
        # self._q_list_nx = self.q_list_next()
        # self._epsilon = self._state_local_copy['epsilon']
        # self._sigma = self._state_local_copy['sigma']
        # self._grid_list = self._state_local_copy['phase_space'].build_gridpoint(self._state_local_copy['npixels'], self._state_local_copy['boxsize'],
        #                                                        self._state_local_copy['DIM'])


    # def qnp_list_in(self, filename):
    #
    #     # _q_list_in, _p_list_in = self._state['phase_space'].read(filename, nsamples=self._state['nsamples_label'])
    #
    #     q_list = [[[-0.62068786, - 0.77235929]]]
    #     p_list = [[[0, 0]]]
    #
    #     _q_list_in, _p_list_in = torch.tensor([q_list, p_list],dtype=torch.float64)
    #
    #     return _q_list_in, _p_list_in

    # give comment to explain
    # def q_list_next(self):
    #
    #     self._state_local_copy['phase_space'].set_q(self._q_list_in)
    #     self._state_local_copy['phase_space'].set_p(self._p_list_in)
    #
    #     self._state_local_copy['nsamples_cur'] = self._state_local_copy['nsamples_label']  # for one step
    #     self._state_local_copy['tau_cur'] = self._state_local_copy['tau_short']  # tau = 0.01
    #     self._state_local_copy['MD_iterations'] = int(self._state_local_copy['tau_short'] / self._state_local_copy['tau_cur'])  # for one step
    #
    #     print('short time step {}, iteration {}'.format(self._state_local_copy['tau_cur'], self._state_local_copy['MD_iterations']))
    #     _q_list_nx, _ = self._linear_itegrator(**self._state_local_copy).integrate(self._NoML_hamiltonian)
    #     _q_list_nx = _q_list_nx[-1].type(torch.float64) # only take the last from the list
    #
    #     return _q_list_nx

    def show_grid_nparticles(self, q_list, title):

        plt.title(title)
        plt.plot(self._grid_list[:,0], self._grid_list[:,1], marker='.', color='k', linestyle='none', markersize=12)
        plt.plot(q_list[:,:, 0], q_list[:,:, 1], marker='x', color='r', linestyle='none', markersize=12)
        plt.show()
        plt.close()

    def phi_max_min(self, phi_field):

        # minm = - ( self._state_local_copy['nparticle'] + 1 )
        # minm = torch.max(phi_field[0])  # phi_field shape : nsamples x npixels, here take one sample that is index=0
        maxm = torch.max(phi_field[0])
        # print('min, max',minm, maxm)

        # dij = 0.1 # q_ij = 0.1
        # maxm_cut = 4 * self._epsilon * ((pow(self._sigma, 12) / pow(dij, 12)) - (pow(self._sigma, 6) / pow(dij, 6)))

        if maxm > self._maxmcut:
            print('phi maxm too big so that cut maxm={:.4f} at dij={}'.format(maxm_cut, dij))
            max_idx = torch.argmax(phi_field[0])
            phi_field[0][max_idx] = maxm_cut  # cannot do that !! keep the phi field. this just show image
            maxm = maxm_cut
            here wrong

        return self._mincut, maxm

    def phi_field(self,phase_space,bc):

        self._phi_field = self._lennard_jones.phi_npixels(phase_space, bc)
        self._phi_field = self._phi_field.reshape((-1, npixels, npixels))

        return self._phi_field

    # def phi_field_next(self):
    #
    #     self._state_local_copy['phase_space'].set_q(self._q_list_nx)
    #     self._state_local_copy['phase_space'].set_grid(self._grid_list)
    #
    #     self._phi_field_nx = self._lennard_jones.phi_npixels(self._state_local_copy['phase_space'], self._state_local_copy['pb_q'])
    #     self._phi_field_nx = self._phi_field_nx.reshape((-1, self._state_local_copy['npixels'], self._state_local_copy['npixels']))
    #
    #     return self._phi_field_nx

    def show_gridimg(self, phi_field):

        minm, maxm = self.phi_max_min(phi_field)
        norm_phi_field = (phi_field - minm) * 255 / (maxm - minm)

        # norm_phi_field = norm_phi_field.reshape((self._state['npixels'], self._state['npixels']))
        plt.imshow(norm_phi_field[0], cmap='gray') # one sample
        plt.colorbar()
        plt.clim(0, 255)
        plt.show()
        plt.close()
