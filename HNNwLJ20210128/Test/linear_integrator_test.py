from HNNwLJ.hamiltonian import hamiltonian
from HNNwLJ.hamiltonian.kinetic_energy import kinetic_energy
from HNNwLJ.hamiltonian.lennard_jones import lennard_jones
from HNNwLJ.hamiltonian.LJ_term import LJ_term
from HNNwLJ.hamiltonian.pb import pb
from HNNwLJ.phase_space.phase_space import phase_space
import torch
from HNNwLJ.integrator import linear_integrator
from HNNwLJ.integrator.methods import linear_velocity_verlet
import matplotlib.pyplot as plt

if __name__ == '__main__':

    nsamples_label = 1
    nsamples_ML = 1
    nparticle = 2
    DIM = 2
    mass = 1
    epsilon = 1.
    sigma = 1.
    boxsize = 8.
    iterations = 10 # 10 steps to pair with large time step = 0.1
    tau_long = 0.5 # short time step for label
    tau_short = 0.01

    # q_list = [[[2,1],[5,4],[1,3]]]
    # p_list = [[[0.1,0.1],[0.2,0.2],[0.2,0.2]]]
    q_list_ = [[[3,2],[2.2,1.21]]]
    p_list_ = [[[0.,0.],[0.,0.]]]

    q_list_tensor, p_list_tensor = torch.tensor([q_list_,p_list_])

    print('initial ', q_list_tensor, p_list_tensor)

    plt.cla()
    plt.xlim(-boxsize / 2, boxsize / 2)
    plt.ylim(-boxsize / 2, boxsize / 2)
    # plt.title(r'boxSize={:.2f}'.format(boxsize), fontsize=15)
    for i in range(nparticle):
        plt.plot(q_list_tensor[:, i, 0], q_list_tensor[:, i, 1], 'o', markersize=15)
        print(q_list_tensor[:, i, 0], q_list_tensor[:, i, 1])

    # plt.show()
    plt.savefig(r'nparticle{}_f1_pos.png'.format(nparticle))
    plt.close()

    tensor_phase_space = phase_space()
    pb = pb()

    NoML_hamiltonian = hamiltonian()
    lj_term = LJ_term(epsilon=epsilon,sigma=sigma,boxsize=boxsize)
    NoML_hamiltonian.append(lennard_jones(lj_term, boxsize))
    NoML_hamiltonian.append(kinetic_energy(mass))

    state = {

        'nsamples_cur': 0,
        'nsamples_label': nsamples_label,
        'nsamples_ML': nsamples_ML,
        'nparticle' : nparticle,
        'DIM' : DIM,
        'boxsize' : boxsize,
        'iterations' : iterations,
        'tau_cur': 0.0,
        'tau_long': tau_long,  # for MD
        'tau_short': tau_short,  # for label (gold standard)
        'integrator_method' : linear_velocity_verlet,
        'phase_space' : tensor_phase_space,
        'pb_q' : pb

        }

    # # initial state
    # state['phase_space'].set_q(q_list_tensor)
    # state['phase_space'].set_p(p_list_tensor)
    #
    # # tau = short time step 0.01 and 10 steps to pair with large time step 0.1
    # state['nsamples_cur'] = state['nsamples_label']
    # state['tau_cur'] = state['tau_short']
    # state['MD_iterations'] = int(state['tau_long'] / state['tau_cur'])
    #
    # q_list_label, p_list_label = linear_integrator(**state).integrate(NoML_hamiltonian)
    #
    # print('label',q_list_label, p_list_label)


    # initial state
    state['phase_space'].set_q(q_list_tensor)
    state['phase_space'].set_p(p_list_tensor)

    # tau = large time step 0.1 and 1 step
    state['nsamples_cur'] = state['nsamples_ML']
    state['tau_cur'] = state['tau_long']  # tau = 0.1
    state['MD_iterations'] = int(state['tau_long'] / state['tau_cur'])

    print('large time step {}'.format(state['tau_cur']))
    print('test', state)
    q_list, p_list = linear_integrator(**state).integrate(NoML_hamiltonian)

    print('noML vv',q_list, p_list)


    plt.cla()
    plt.xlim(-boxsize / 2, boxsize / 2)
    plt.ylim(-boxsize / 2, boxsize / 2)
    # plt.title(r'boxSize={:.2f}'.format(boxsize), fontsize=15)
    for i in range(nparticle):
        plt.plot(q_list[:, i, 0], q_list[:, i, 1], 'o', markersize=15)
        print(q_list[:, i, 0], q_list[:, i, 1])

    # plt.show()
    plt.savefig(r'nparticle{}_f2_pos.png'.format(nparticle))
    plt.close()