import numpy as np
from HNNwLJ.hamiltonian import hamiltonian
from HNNwLJ.hamiltonian.kinetic_energy import kinetic_energy
from HNNwLJ.hamiltonian.lennard_jones import lennard_jones
from HNNwLJ.hamiltonian.LJ_term import LJ_term
from HNNwLJ.hamiltonian.pb import pb
from HNNwLJ.phase_space.phase_space import phase_space
from HNNwLJ.optical_flow_HNN.ljbox2gridimg import ljbox2gridimg
from HNNwLJ.optical_flow_HNN.optical_flow2img import optical_flow2img
from HNNwLJ.integrator import linear_integrator
from HNNwLJ.integrator.methods import linear_velocity_verlet
import matplotlib.pyplot as plt

if __name__ == '__main__':

    mass = 1
    epsilon = 1.
    sigma = 1.
    boxsize = 6.
    DIM = 2
    nsamples = 1
    npixels = 16

    nsamples_label = 1
    nsamples_ML = 1
    nparticle = 2
    iterations = 10 # 10 steps to pair with large time step = 0.1
    tau_long = 0.5 # short time step for label
    tau_short = 0.001

    NoML_hamiltonian = hamiltonian()
    lj_term = LJ_term(epsilon = epsilon, sigma = sigma, boxsize = boxsize)
    NoML_hamiltonian.append(lennard_jones(lj_term, boxsize))
    NoML_hamiltonian.append(kinetic_energy(mass))

    phase_space = phase_space()
    pb = pb()

    state = {

        'nsamples_cur': 0,
        'nsamples_label': nsamples_label,
        'nsamples_ML': nsamples_ML,
        'nparticle' : nparticle,
        'npixels' : npixels,
        'epsilon' : 1.,
        'sigma' : 1.,
        'DIM' : DIM,
        'boxsize' : boxsize,
        'iterations' : iterations,
        'MD_iterations': 0,
        'tau_cur': 0.0,
        'tau_long': tau_long,  # for MD
        'tau_short': tau_short,  # for label (gold standard)
        'integrator_method' : linear_velocity_verlet,
        'phase_space' : phase_space,
        'pb_q' : pb

        }

    # load filename
    filename = '../init_config/N_particle{}_samples{}_rho0.1_T0.04_pos_sampled.pt'.format(nparticle, nsamples_label)

    phi_field = ljbox2gridimg(NoML_hamiltonian, lennard_jones(lj_term, boxsize), linear_integrator, filename, **state)

    q_list_in = phi_field._q_list_in
    # phi_field.show_grid_nparticles(q_list_in, 'phi field(t)')

    q_list_nx = phi_field._q_list_nx
    # phi_field.show_grid_nparticles(q_list_nx, 'phi field(t+dt)')

    phi_field_initial = phi_field.phi_field_initial()
    # phi_field.show_gridimg(phi_field_initial)

    phi_field_next = phi_field.phi_field_next()
    # phi_field.show_gridimg(phi_field_next)

    optical_flow2img = optical_flow2img(phi_field_initial, phi_field_next)
    flow_vectors = optical_flow2img.p_field()
    optical_flow2img.visualize_flow_file(flow_vectors)

