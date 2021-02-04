import sys
import os
sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("../parameters"))

from HNN.pair_wise_HNN import pair_wise_HNN
from phase_space.phase_space import phase_space
from integrator import linear_integrator
from parameters.MD_paramaters import MD_parameters
from HNN.models import pair_wise_MLP
import torch
import matplotlib.pyplot as plt


if __name__ == '__main__':

    # q_list = [[[3,2],[2.2,1.21]]]
    # p_list = [[[0.1,0.1],[0.1,0.1]]]
    # q_list = [[[2.3945972, 0.79560974], [1.29235072, 0.64889931], [1.66907468, 1.693532]]]
    # p_list = [[[0.1,0.],[0.,0.4],[0.1, 0.3]]]
    # q_list = [[[-0.62068786, - 0.77235929],[1.23484839, - 1.33486261],[0.12320894, - 1.58505487],[0.42893553, - 0.5222273]]]
    # p_list = [[[0,0],[0,0],[0,0],[0,0]]]

    # q_list, p_list = torch.tensor([q_list,p_list])

    gen_nsamples = MD_parameters.gen_nsamples
    nsamples = MD_parameters.nsamples
    mass = MD_parameters.mass
    boxsize = MD_parameters.boxsize
    nparticle = MD_parameters.nparticle
    DIM = MD_parameters.DIM

    phase_space = phase_space()

    pair_wise_HNN_obj = pair_wise_HNN(pair_wise_MLP())
    linear_integrator_obj = linear_integrator(MD_parameters.integrator_method)

    filename = '../init_config/N_particle{}_samples{}_rho0.1_T0.04_pos_sampled.pt'.format(nparticle, gen_nsamples)

    _phase_space = torch.load(filename)
    q_list, p_list = _phase_space[0][:MD_parameters.nsamples], _phase_space[1][:MD_parameters.nsamples]

    # print('print phase spase helper', tensor_phase_space.helper())
    noMLhamiltonian = super(type(pair_wise_HNN_obj), pair_wise_HNN_obj)

    # print(noMLhamiltonian.hi())
    terms = noMLhamiltonian.get_terms()

    phase_space.set_q(q_list)
    phase_space.set_p(p_list)
    k_term = terms[1].energy(phase_space)
    print('kinetic energy',k_term)

    phase_space.set_q(q_list)
    u_term = terms[0].energy(phase_space)
    print('potential energy',u_term)

    plt.title('T={}'.format(MD_parameters.temp),fontsize=15)
    plt.plot(u_term,'k-')
    plt.xlabel('mcs',fontsize=20)
    plt.ylabel(r'$U_{ij}$',fontsize=20)
    plt.show()

    phase_space.set_q(q_list)
    energy = noMLhamiltonian.total_energy(phase_space)

    print('total energy',energy)
    # numpy_energy = hamiltonian.total_energy(numpy_phase_space) #- cannot run

    quit()
    phase_space.set_q(q_list)
    phase_space.set_p(p_list)

    tensor_dHdq = noMLhamiltonian.dHdq(phase_space)
    print('dHdq',tensor_dHdq)
    #numpy_dHdq = hamiltonian.dHdq(numpy_phase_space,pb)

    phase_space.set_q(q_list)
    phase_space.set_p(p_list)

    tensor_d2Hdq2 = noMLhamiltonian.d2Hdq2(phase_space)
    print('d2Hdq2',tensor_d2Hdq2)
