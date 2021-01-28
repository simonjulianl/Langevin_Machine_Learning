from HNNwLJ20210128.hamiltonian import hamiltonian
from HNNwLJ20210128.hamiltonian.kinetic_energy import kinetic_energy
from HNNwLJ20210128.hamiltonian.lennard_jones import lennard_jones
from HNNwLJ20210128.hamiltonian.LJ_term import LJ_term
from HNNwLJ20210128.phase_space.phase_space import phase_space
from HNNwLJ20210128.parameters.MD_paramaters import MD_parameters
import torch


if __name__ == '__main__':

    q_list = [[[3,2],[2.2,1.21]]]
    p_list = [[[0.1,0.1],[0.1,0.1]]]
    # q_list = [[[2.3945972, 0.79560974], [1.29235072, 0.64889931], [1.66907468, 1.693532]]]
    # p_list = [[[0.1,0.],[0.,0.4],[0.1, 0.3]]]
    # q_list = [[[-0.62068786, - 0.77235929],[1.23484839, - 1.33486261],[0.12320894, - 1.58505487],[0.42893553, - 0.5222273]]]
    # p_list = [[[0,0],[0,0],[0,0],[0,0]]]

    q_list_tensor, p_list_tensor = torch.tensor([q_list,p_list])

    # q_list_numpy, p_list_numpy = np.array([q_list,p_list])
    # numpy_phase_space = phase_space()
    # numpy_phase_space.set_q(q_list_numpy)
    # numpy_phase_space.set_p(p_list_numpy)

    mass = MD_parameters.mass
    epsilon = MD_parameters.epsilon
    sigma = MD_parameters.sigma
    boxsize = MD_parameters.boxsize
    nparticle = MD_parameters.nparticle
    DIM = MD_parameters.DIM

    tensor_phase_space = phase_space()
    # print('print phase spase helper', tensor_phase_space.helper())

    hamiltonian = hamiltonian()
    lj_term = LJ_term(epsilon = epsilon, sigma = sigma, boxsize = boxsize)
    hamiltonian.append(lennard_jones(lj_term, boxsize))
    hamiltonian.append(kinetic_energy(mass))

    tensor_phase_space.set_q(q_list_tensor)
    tensor_phase_space.set_p(p_list_tensor)

    tensor_energy = hamiltonian.total_energy(tensor_phase_space)
    print('total energy',tensor_energy)
    # numpy_energy = hamiltonian.total_energy(numpy_phase_space) #- cannot run

    tensor_phase_space.set_q(q_list_tensor)
    tensor_phase_space.set_p(p_list_tensor)

    tensor_dHdq = hamiltonian.dHdq(tensor_phase_space)
    print('dHdq',tensor_dHdq)
    #numpy_dHdq = hamiltonian.dHdq(numpy_phase_space,pb)

    tensor_phase_space.set_q(q_list_tensor)
    tensor_phase_space.set_p(p_list_tensor)

    tensor_d2Hdq2 = hamiltonian.d2Hdq2(tensor_phase_space)
    print('d2Hdq2',tensor_d2Hdq2)
