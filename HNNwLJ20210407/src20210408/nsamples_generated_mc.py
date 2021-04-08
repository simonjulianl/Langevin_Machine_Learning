from phase_space.phase_space         import phase_space
from integrator.linear_integrator    import linear_integrator
from integrator.methods              import linear_velocity_verlet
from hamiltonian.noML_hamiltonian    import noML_hamiltonian
from utils.data_io                   import data_io

import sys
import torch
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

if __name__ == '__main__':

    ''' potential energy fluctuations for mc steps that appended to nsamples and 
        histogram of nsamples generated from mc simulation'''

    # run something like this
    # python nsamples_generated_mc.py 4 0.04 ../data/init_config/n4train/filename ../data/init_config/n4valid/filename

    argv = sys.argv

    npartilce = argv[1]
    temp      = argv[2]
    infile1   = argv[3]
    infile2   = argv[4]

    # parameters
    nparticle    = npartilce
    temp         = temp

    phase_space = phase_space()
    hamiltonian_obj = noML_hamiltonian()
    linear_integrator_obj = linear_integrator(linear_velocity_verlet.linear_velocity_verlet, linear_velocity_verlet.linear_velocity_verlet_backward)

    terms = hamiltonian_obj.get_terms()

    # nsamples from mc step generated for training data
    init_qp_train, _, _, boxsize = data_io.read_trajectory_qp(infile1)
    # init_qp_train.shape = [nsamples, (q, p), 1, nparticle, DIM]

    init_q_train = torch.squeeze(init_qp_train[:,0,:,:,:], dim=1)
    init_p_train = torch.squeeze(init_qp_train[:,1,:,:,:], dim=1)

    phase_space.set_q(init_q_train)
    phase_space.set_p(init_p_train)
    phase_space.set_boxsize(boxsize)

    u_term1 = terms[0].energy(phase_space)

    # nsamples from mc step generated for valid data
    init_qp_valid, _, _, _ = data_io.read_trajectory_qp(infile2)

    init_q_valid = torch.squeeze(init_qp_valid[:,0,:,:,:], dim=1)
    init_p_valid = torch.squeeze(init_qp_valid[:,1,:,:,:], dim=1)

    phase_space.set_q(init_q_valid)
    phase_space.set_p(init_p_valid)

    u_term2 = terms[0].energy(phase_space)

    plt.title('mc steps appended to nsamples given at T={}'.format(temp),fontsize=15)
    plt.plot(u_term1,'k-', label = 'potential energy using {} samples for training'.format(init_q_train.shape[0]))
    plt.plot(u_term2,'r-', label = 'potential energy using {} samples for validation'.format(init_q_valid.shape[0]))
    plt.xlabel('mcs',fontsize=20)
    plt.ylabel(r'$U_{ij}$',fontsize=20)
    plt.legend()
    plt.show()

    # plt.xlim(xmin=-5.1, xmax = -4.3)
    fig, ax = plt.subplots()
    plt.hist(u_term1.numpy(), bins=100, color='k', alpha = 0.5, label = 'train: histogram of {} samples'.format(init_q_train.shape[0]))
    plt.hist(u_term2.numpy(), bins=100, color='r', alpha = 0.5, label = 'valid: histogram of {} samples'.format(init_q_valid.shape[0]))
    plt.xlabel(r'$U_{ij}$',fontsize=20)
    plt.ylabel('hist', fontsize=20)
    anchored_text = AnchoredText('nparticle={} boxsize={:.3f} temp={} data split = train {} / valid {}'.format(nparticle, boxsize, temp, init_q_train.shape[0], init_q_valid.shape[0]), loc= 'upper left', prop=dict(fontweight="normal", size=12))
    ax.add_artist(anchored_text)

    plt.legend()
    plt.show()

