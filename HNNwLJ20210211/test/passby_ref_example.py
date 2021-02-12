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
import torch
import copy

class A:
    def __init__(self,**state):
        self._state_local_copy = copy.deepcopy(state)

    def f(self):
        q_list = [[[1, 1]]]
        p_list = [[[1, 1]]]

        _q_list_in, _p_list_in = torch.tensor([q_list, p_list], dtype=torch.float64)

        self._state_local_copy['phase_space'].set_q(_q_list_in)
        self._state_local_copy['phase_space'].set_p(_p_list_in)
        return

#======================================================

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
    nparticle = 1
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
    print('phase space ',state['phase_space'])

    q_list = [[[-0.62068786, - 0.77235929]]]
    p_list = [[[0, 0]]]

    _q_list_in, _p_list_in = torch.tensor([q_list, p_list], dtype=torch.float64)

    state['phase_space'].set_q(_q_list_in)
    state['phase_space'].set_p(_p_list_in)
    print('before phase space q', state['phase_space'].get_q())
    print('brefore phase space p', state['phase_space'].get_p())

    a = A(**state)

    a.f()

    print('after phase space q', state['phase_space'].get_q())
    print('after phase space p', state['phase_space'].get_p())
