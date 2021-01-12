from HNNwLJ.hamiltonian import hamiltonian
from HNNwLJ.hamiltonian.kinetic_energy import kinetic_energy
from HNNwLJ.hamiltonian.lennard_jones import lennard_jones
from HNNwLJ.hamiltonian.LJ_term import LJ_term
from HNNwLJ.hamiltonian.pb import pb
from HNNwLJ.phase_space.phase_space import phase_space
import torch
from HNNwLJ.integrator import linear_integrator
from HNNwLJ.integrator.methods import linear_velocity_verlet

if __name__ == '__main__':

    nsample = 1
    N_particle = 2
    DIM = 2
    mass = 1
    epsilon = 1.
    sigma = 1.
    boxsize = 6.
    iterations = 10 # 10 steps to pair with large time step = 0.1
    tau = 0.01 # short time step for label

    # q_list = [[[2,1],[5,4],[1,3]]]
    # p_list = [[[0.1,0.1],[0.2,0.2],[0.2,0.2]]]
    q_list = [[[3,2],[2.2,1.21]]]
    p_list = [[[0.,0.],[0.,0.]]]

    q_list_tensor, p_list_tensor = torch.tensor([q_list,p_list])
    tensor_phase_space = phase_space()
    pb = pb()

    NoML_hamiltonian = hamiltonian()
    lj_term = LJ_term(epsilon=epsilon,sigma=sigma,boxsize=boxsize)
    NoML_hamiltonian.append(lennard_jones(lj_term, boxsize))
    NoML_hamiltonian.append(kinetic_energy(mass))

    state = {

        'N' : nsample,
        'particle' : N_particle,
        'DIM' : DIM,
        'BoxSize' : boxsize,
        'iterations' : iterations,
        'tau' : tau,
        'integrator_method' : linear_velocity_verlet,
        'phase_space' : tensor_phase_space,
        'pb_q' : pb

        }

    # initial state
    state['phase_space'].set_q(q_list_tensor)
    state['phase_space'].set_p(p_list_tensor)

    # tau = short time step 0.01 and 10 steps to pair with large time step 0.1
    print('short time step {}'.format(state['tau']))
    print('test', state)
    q_list_label, p_list_label = linear_integrator(**state).integrate(NoML_hamiltonian)

    print('label',q_list_label, p_list_label)

    # initial state
    state['phase_space'].set_q(q_list_tensor)
    state['phase_space'].set_p(p_list_tensor)

    # tau = large time step 0.1 and 1 step
    state['tau'] = state['tau'] * state['iterations']   # tau = 0.1
    state['iterations'] = int(state['tau'] * state['iterations']) # 1 step
    print('large time step {}'.format(state['tau']))
    print('test', state)
    q_list, p_list = linear_integrator(**state).integrate(NoML_hamiltonian)

    print('noML vv',q_list, p_list)
