import sys
import os
sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("../parameters"))

from HNN.pair_wise_HNN import pair_wise_HNN
from phase_space.phase_space import phase_space
from integrator import linear_integrator
from parameters.MC_paramaters import MC_parameters
from parameters.MD_paramaters import MD_parameters
from HNN.models import pair_wise_MLP
import torch
import matplotlib.pyplot as plt


if __name__ == '__main__':

    gen_nsamples = MD_parameters.gen_nsamples
    nsamples = MD_parameters.nsamples
    mass = MD_parameters.mass
    boxsize = MD_parameters.boxsize
    nparticle = MD_parameters.nparticle
    DIM = MD_parameters.DIM

    phase_space = phase_space()

    pair_wise_HNN_obj = pair_wise_HNN(pair_wise_MLP())
    linear_integrator_obj = linear_integrator(MD_parameters.integrator_method)
    noMLhamiltonian = super(type(pair_wise_HNN_obj), pair_wise_HNN_obj)

    terms = noMLhamiltonian.get_terms()

    # 1000 samples from 1000 mcs
    filename1 = '../init_config/N_particle{}_new_step{}_rho0.1_T0.04_pos_sampled.pt'.format(nparticle, nsamples)
    q_list1, p_list1 = torch.load(filename1)

    phase_space.set_p(p_list1)
    phase_space.set_q(q_list1)
    u_term1 = terms[0].energy(phase_space)
    # print('potential energy',u_term1)

    # 1000 samples from 20000 mcs
    filename2 = '../init_config/N_particle{}_new_step{}_rho0.1_T0.04_pos_sampled.pt'.format(nparticle, gen_nsamples)
    q_list2, p_list2 = torch.load(filename2)

    # shuffle
    g = torch.Generator()
    g.manual_seed(MD_parameters.seed)

    idx = torch.randperm(q_list2.shape[0], generator=g)

    q_list_shuffle_ = q_list2[idx]
    p_list_shuffle_ = p_list2[idx]

    q_list_shuffle = q_list_shuffle_[:MD_parameters.nsamples]
    p_list_shuffle = p_list_shuffle_[:MD_parameters.nsamples]

    phase_space.set_p(p_list_shuffle)
    phase_space.set_q(q_list_shuffle)
    u_term2 = terms[0].energy(phase_space)
    # print('potential energy',u_term2)

    plt.title('T={}'.format(MC_parameters.temperature),fontsize=15)
    plt.plot(u_term1,'k-', label = '{} samples from {} mcs'.format(nsamples, nsamples))
    plt.plot(u_term2,'r-', label = '{} samples from {} mcs'.format(nsamples, gen_nsamples))
    plt.xlabel('mcs',fontsize=20)
    plt.ylabel(r'$U_{ij}$',fontsize=20)
    plt.legend()
    plt.show()

    plt.xlim(xmin=-5.1, xmax = -4.3)
    plt.hist(u_term1.numpy(), bins=100, color='k', alpha = 0.5, label = '{} samples from {} mcs'.format(nsamples, nsamples))
    plt.hist(u_term2.numpy(), bins=100, color='r', alpha = 0.5, label = '{} samples from {} mcs'.format(nsamples,gen_nsamples))
    plt.xlabel(r'$U_{ij}$',fontsize=20)
    plt.ylabel('hist', fontsize=20)
    plt.legend()
    plt.show()

