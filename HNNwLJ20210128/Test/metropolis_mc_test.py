import torch
import math
from HNNwLJ20210128.integrator.metropolis_mc import metropolis_mc
from HNNwLJ20210128.phase_space.phase_space import phase_space
from HNNwLJ20210128.hamiltonian import hamiltonian
from HNNwLJ20210128.hamiltonian.LJ_term import LJ_term
from HNNwLJ20210128.hamiltonian.lennard_jones import lennard_jones
from HNNwLJ20210128.hamiltonian.kinetic_energy import kinetic_energy

if __name__ == '__main__':

    seed = 23645
    nsamples = 1
    nparticle = 2
    DIM = 2
    mass = 1
    epsilon = 1.
    sigma = 1.
    rho = 0.1
    T = 0.04
    dq = 0.1
    boxsize = math.sqrt(nparticle / rho)

    phase_space = phase_space()
    pb = pb()

    noML_hamiltonian = hamiltonian()
    lj_term = LJ_term(epsilon = epsilon, sigma = sigma, boxsize = boxsize)
    noML_hamiltonian.append(lennard_jones(lj_term, boxsize))
    noML_hamiltonian.append(kinetic_energy(mass))

    state = {
        'seed': seed,
        'kB': 1.0,  # put as a constant
        'temperature': T,
        'DIM': DIM,
        'm': mass,
        'nsamples': nsamples,  # for train 50000 for test 5000 each temp
        'nparticle': nparticle,
        'boxsize': boxsize,
        'phase_space': phase_space,
        'pb_q': pb,
        'iterations': 3,  # for test 44000 #for train 224000 624000 62440
        'DISCARD': 1,  # for test 24000 #for train 24000  622000 62400
        'dq': dq
    }

    mc = metropolis_mc( noML_hamiltonian, **state)
    mc.integrate()