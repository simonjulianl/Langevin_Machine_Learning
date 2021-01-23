import numpy as np
from HNNwLJ.hamiltonian import hamiltonian
from HNNwLJ.hamiltonian.kinetic_energy import kinetic_energy
from HNNwLJ.hamiltonian.lennard_jones import lennard_jones
from HNNwLJ.hamiltonian.LJ_term import LJ_term
from HNNwLJ.hamiltonian.pb import pb
from HNNwLJ.phase_space.phase_space import phase_space
import torch
from HNNwLJ.optical_flow_HNN.dataset import dataset
from HNNwLJ.integrator import linear_integrator
from HNNwLJ.integrator.methods import linear_velocity_verlet
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # q_list = [[[-0.62068786, - 0.77235929],[1.23484839, - 1.33486261]]]
    # p_list = [[[0, 0],[0, 0]]]
    #
    # q_list_tensor, p_list_tensor = torch.tensor([q_list, p_list])
    # print('noML inital state',q_list_tensor, p_list_tensor)

    phase_space = phase_space()
    pb = pb()

    mass = 1
    epsilon = 1.
    sigma = 1.
    boxsize = 6.
    DIM = 2
    nsamples = 1
    npixels = 32

    nsamples_label = 1
    nsamples_ML = 1
    nparticle = 2
    iterations = 10 # 10 steps to pair with large time step = 0.1
    tau_long = 0.5 # short time step for label
    tau_short = 0.01

    NoML_hamiltonian = hamiltonian()
    lj_term = LJ_term(epsilon = epsilon, sigma = sigma, boxsize = boxsize)
    NoML_hamiltonian.append(lennard_jones(lj_term, boxsize))
    NoML_hamiltonian.append(kinetic_energy(mass))

    state = {

        'nsamples_cur': 0,
        'nsamples_label': nsamples_label,
        'nsamples_ML': nsamples_ML,
        'nparticle' : nparticle,
        'npixels' : npixels,
        'DIM' : DIM,
        'boxsize' : boxsize,
        'iterations' : iterations,
        'tau_cur': 0.0,
        'tau_long': tau_long,  # for MD
        'tau_short': tau_short,  # for label (gold standard)
        'integrator_method' : linear_velocity_verlet,
        'phase_space' : phase_space,
        'pb_q' : pb

        }

    filename = '../init_config/N_particle{}_samples{}_rho0.1_T0.04_pos_sampled.pt'.format(nparticle, nsamples_label)
    phi_field = dataset(NoML_hamiltonian, linear_integrator, lennard_jones(lj_term, boxsize), epsilon, sigma, **state)
    phi_field_initial = phi_field.phi_field_initial(filename =filename)
    phi_field_next = phi_field.phi_field_next(filename=filename)
    phi_field.show_img(phi_field_initial)
    phi_field.show_img(phi_field_next)
    quit()
    lj_gird = lennard_jones(lj_term, boxsize)
    grid_list = phase_space.build_gridpoint(npixels, boxsize, DIM)
    print(grid_list.shape)

    for i in range(npixels):
        for j in range(npixels):

            plt.plot(grid_list[:,0], grid_list[:,1], marker='.', color='k', linestyle='none', markersize=12)

    plt.plot(q_list_tensor[:,:, 0], q_list_tensor[:,:, 1], marker='x', color='r', linestyle='none', markersize=12)
    # plt.show()

    phase_space.set_q(q_list_tensor)
    phase_space.set_grid(grid_list)

    def max_min(potential_grid):

        minm = potential_grid[0].min()
        maxm = potential_grid[0].max()

        dij = 0.4  # q_ij = 0.3
        maxm_cut = 4 * epsilon * ((pow(sigma, 12) / pow(dij, 12)) - (pow(sigma, 6) / pow(dij, 6)))

        if maxm > maxm_cut:
            max_idx = torch.argmax(potential_grid[0])
            potential_grid[0][max_idx] = maxm_cut
            maxm = maxm_cut

        return minm, maxm

    potential_grid = lj_gird.phi_npixels(phase_space, pb)
    print('inital',potential_grid)
    minm, maxm = max_min(potential_grid)
    print(minm, maxm )

    # normalize img to 0-255
    # to show img
    norm_potenital = (potential_grid - minm) * 255 / (maxm - minm )
    norm_potenital = norm_potenital.reshape((npixels,npixels))
    plt.imshow(norm_potenital, cmap='gray')
    plt.colorbar()
    plt.clim(0, 255)
    # plt.show()

    # initial state
    phase_space.set_q(q_list_tensor)
    phase_space.set_p(p_list_tensor)

    # tau = short time step
    state['nsamples_cur'] = state['nsamples_ML'] # for one step
    state['tau_cur'] = state['tau_short']  # tau = 0.01
    state['MD_iterations'] = int(state['tau_short'] / state['tau_cur'])

    print('short time step {}'.format(state['tau_cur']))
    q_next_list, p_next_list = linear_integrator(**state).integrate(NoML_hamiltonian)

    print('noML next state',q_next_list, p_next_list)

    phase_space.set_q(q_next_list)
    phase_space.set_p(p_next_list)
    phase_space.set_grid(grid_list)

    potential_grid = lj_gird.phi_npixels(phase_space, pb)
    print('next',potential_grid)
    minm, maxm = max_min(potential_grid)
    print(minm, maxm )

    norm_potenital = (potential_grid - minm) * 255 / (maxm - minm )
    norm_potenital = norm_potenital.reshape((npixels,npixels))
    plt.imshow(norm_potenital, cmap='gray')
    plt.colorbar()
    plt.clim(0, 255)
    # plt.show()

