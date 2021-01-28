from HNNwLJ20210128.hamiltonian import hamiltonian
from HNNwLJ20210128.hamiltonian.kinetic_energy import kinetic_energy
from HNNwLJ20210128.hamiltonian.lennard_jones import lennard_jones
from HNNwLJ20210128.hamiltonian.LJ_term import LJ_term
from HNNwLJ20210128.phase_space.phase_space import phase_space
from HNNwLJ20210128.integrator import linear_integrator
from HNNwLJ20210128.integrator.methods import linear_velocity_verlet
from HNNwLJ20210128.parameters.MD_paramaters import MD_parameters
import matplotlib.pyplot as plt
import torch

if __name__ == '__main__':

    nsamples_cur = MD_parameters.nsamples_cur
    nsamples_label = MD_parameters.nsamples_label
    nsamples_ML = MD_parameters.nsamples_ML
    mass = MD_parameters.mass
    epsilon = MD_parameters.epsilon
    sigma = MD_parameters.sigma
    boxsize = MD_parameters.boxsize
    nparticle = MD_parameters.nparticle
    DIM = MD_parameters.DIM
    MD_iterations = MD_parameters.MD_iterations # 10 steps to pair with large time step = 0.1
    tau_cur = MD_parameters.tau_cur
    tau_long = MD_parameters.tau_long # short time step for label
    tau_short = MD_parameters.tau_short

    # q_list = [[[2,1],[5,4],[1,3]]]
    # p_list = [[[0.1,0.1],[0.2,0.2],[0.2,0.2]]]
    q_list_ = [[[3,2],[2.2,1.21]]]
    p_list_ = [[[0.,0.],[0.,0.]]]

    q_list_tensor, p_list_tensor = torch.tensor([q_list_,p_list_])

    print('initial ', q_list_tensor, p_list_tensor)

    phase_space = phase_space()

    NoML_hamiltonian = hamiltonian()
    lj_term = LJ_term(epsilon=epsilon,sigma=sigma,boxsize=boxsize)
    NoML_hamiltonian.append(lennard_jones(lj_term, boxsize))
    NoML_hamiltonian.append(kinetic_energy(mass))

    integrator_method = linear_velocity_verlet

    # initial state
    phase_space.set_q(q_list_tensor)
    phase_space.set_p(p_list_tensor)

    # tau = short time step 0.01 and 10 steps to pair with large time step 0.1
    nsamples_cur = nsamples_label
    tau_cur = tau_short
    MD_iterations = int(tau_long / tau_cur)

    print('for label : nsamples_cur, tau_cur, MD_iterations')
    print(nsamples_cur, tau_cur, MD_iterations)
    q_list_label, p_list_label = linear_integrator(NoML_hamiltonian, integrator_method).integrate(phase_space, MD_iterations, nsamples_cur, nparticle, DIM, tau_cur, boxsize)

    print('label',q_list_label.shape, p_list_label.shape)

    # initial state
    phase_space.set_q(q_list_tensor)
    phase_space.set_p(p_list_tensor)

    # tau = large time step 0.1 and 1 step
    nsamples_cur = nsamples_ML
    tau_cur = tau_long  # tau = 0.1
    MD_iterations = int(tau_long / tau_cur)

    print('At large time : nsamples_cur, tau_cur, MD_iterations')
    print(nsamples_cur, tau_cur, MD_iterations)
    q_list, p_list = linear_integrator(NoML_hamiltonian, integrator_method).integrate(phase_space, MD_iterations, nsamples_cur, nparticle, DIM, tau_cur, boxsize)

    print('noML vv',q_list, p_list)
