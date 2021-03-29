import sys
import os
sys.path.append(os.path.abspath("./parameters"))

from HNN.pair_wise_HNN import pair_wise_HNN
from HNN.models import pair_wise_MLP
from phase_space.phase_space import phase_space
from integrator import linear_integrator
from parameters.MC_parameters import MC_parameters
from parameters.MD_parameters import MD_parameters
import torch
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

if __name__ == '__main__':

    nsamples = MD_parameters.nsamples
    mass = MD_parameters.mass
    temperature = MC_parameters.temperature
    boxsize = MD_parameters.boxsize
    nparticle = MD_parameters.nparticle
    DIM = MD_parameters.DIM

    phase_space = phase_space()

    pair_wise_HNN_obj = pair_wise_HNN(pair_wise_MLP())
    linear_integrator_obj = linear_integrator(MD_parameters.integrator_method)
    noMLhamiltonian = super(type(pair_wise_HNN_obj), pair_wise_HNN_obj)

    terms = noMLhamiltonian.get_terms()

    # 1000 samples from 1000 mcs
    filename1 = './init_config/nparticle{}_new_nsim_rho0.1_T{}_pos_train_sampled.pt'.format(nparticle, temperature)
    q_list1, p_list1 = torch.load(filename1)

    phase_space.set_p(p_list1)
    phase_space.set_q(q_list1)
    u_term1 = terms[0].energy(phase_space)
    # print('potential energy',u_term1)

    # 1000 samples from 20000 mcs
    filename2 = './init_config/nparticle{}_new_nsim_rho0.1_T{}_pos_valid_sampled.pt'.format(nparticle, temperature)
    q_list2, p_list2 = torch.load(filename2)

    phase_space.set_p(p_list2)
    phase_space.set_q(q_list2)
    u_term2 = terms[0].energy(phase_space)
    # print('potential energy',u_term2)

    # plt.title('T={}'.format(MC_parameters.temperature),fontsize=15)
    # plt.plot(u_term1,'k-', label = 'train: each {} mcs from 50 random samplings'.format(nsamples))
    # plt.plot(u_term2,'r-', label = 'valid: each {} mcs from 5 random samplings'.format(nsamples))
    # plt.xlabel('mcs',fontsize=20)
    # plt.ylabel(r'$U_{ij}$',fontsize=20)
    # plt.legend()
    # plt.show()

    # plt.xlim(xmin=-5.1, xmax = -4.3)
    fig, ax = plt.subplots()
    plt.hist(u_term1.numpy(), bins=100, color='k', alpha = 0.5, label = 'train: each {} mcs from 80 random samplings'.format(nsamples))
    plt.hist(u_term2.numpy(), bins=100, color='r', alpha = 0.5, label = 'valid: each {} mcs from 8 random samplings'.format(nsamples))
    plt.xlabel(r'$U_{ij}$',fontsize=20)
    plt.ylabel('hist', fontsize=20)
    anchored_text = AnchoredText('nparticle={} boxsize={:.3f} temperature={} data split = train 50000 / valid 500'.format(nparticle, boxsize, temperature), loc= 'center', prop=dict(fontweight="normal", size=12))
    ax.add_artist(anchored_text)

    plt.legend()
    plt.show()

