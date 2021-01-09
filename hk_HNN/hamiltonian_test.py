import hamiltonian
from kinetic_energy import kinetic_energy
from lennard_jones import lennard_jones
from LJ_term import LJ_term
from pb import pb
from phase_space import phase_space
import numpy as np
import torch


if __name__ == '__main__':

    q_list = [[[2,3],[3,4]]]
    p_list = [[[0.1,0.1],[0.2,0.2]]]
    q_list_tensor, p_list_tensor = torch.tensor([q_list,p_list])
    tensor_phase_space = phase_space()
    tensor_phase_space.set_q(q_list_tensor)
    tensor_phase_space.set_p(p_list_tensor)

    q_list_numpy, p_list_numpy = np.array([q_list,p_list])
    numpy_phase_space = phase_space()
    numpy_phase_space.set_q(q_list_numpy)
    numpy_phase_space.set_p(p_list_numpy)

    pb = pb()

    mass = 1
    epsilon = 1.
    sigma = 1.
    boxsize = 2.

    hamiltonian = hamiltonian.hamiltonian()
    lj_term = LJ_term(epsilon=epsilon,sigma=sigma,boxsize=boxsize)
    hamiltonian.append(lennard_jones(lj_term, boxsize))
    hamiltonian.append(kinetic_energy(mass))

    tensor_energy = hamiltonian.total_energy(tensor_phase_space,pb)
    numpy_energy = hamiltonian.total_energy(numpy_phase_space,pb) - cannot run

    tensor_dHdq = hamiltonian.dHdq(tensor_phase_space,pb)
    print(tensor_dHdq)
    #numpy_dHdq = hamiltonian.dHdq(numpy_phase_space,pb)