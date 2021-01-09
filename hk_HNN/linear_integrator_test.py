from hamiltonian import hamiltonian
from kinetic_energy import kinetic_energy
from lennard_jones import lennard_jones
from LJ_term import LJ_term
from pb import pb
from phase_space import phase_space
import torch
from linear_velocity_verlet import linear_velocity_verlet

if __name__ == '__main__':

    q_list = [[[2,1],[5,4],[1,3]]]
    p_list = [[[0.1,0.1],[0.2,0.2],[0.2,0.2]]]
    q_list_tensor, p_list_tensor = torch.tensor([q_list,p_list])
    tensor_phase_space = phase_space()
    pb = pb()

    mass = 1
    epsilon = 1.
    sigma = 1.
    boxsize = 2.
    iterations = 1
    tau = 0.1

    hamiltonian = hamiltonian()
    lj_term = LJ_term(epsilon=epsilon,sigma=sigma,boxsize=boxsize)
    hamiltonian.append(lennard_jones(lj_term, boxsize))
    hamiltonian.append(kinetic_energy(mass))

    state = {

        'hamiltonian' : hamiltonian,
        'iterations' : iterations,
        'BoxSize' : boxsize,
        'tau' : tau,
        'pb_q' : pb,
        'phase_space' : tensor_phase_space
        }

    tensor_phase_space.set_q(q_list_tensor)
    tensor_phase_space.set_p(p_list_tensor)

    for i in range(iterations):

        linear_velocity_verlet(**state)


