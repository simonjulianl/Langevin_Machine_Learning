from phase_space.phase_space         import phase_space
from integrator.linear_integrator    import linear_integrator
from integrator.methods              import linear_velocity_verlet
from parameters.MD_parameters        import MD_parameters

import re
import glob
import math
import torch
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

def keyFunc(afilename):
    '''function to change the sort from 'ASCIIBetical' to numeric by isolating the number
        in the filename'''

    nondigits = re.compile("\D")
    return int(nondigits.sub("", afilename))

if __name__ == '__main__':

    ''' potential energy fluctuations for mc steps and 
        histogram of nsamples generated from mc simulation'''

    # parameters
    new_mcs      = 1   # the number of random sampling for mc
    mcstep       = 2   # mc steps each sample

    temp         = 0.04
    nparticle    = 2
    DIM          = 2
    rho          = 0.1
    boxsize      = math.sqrt(nparticle/rho)

    root_path    = '../data/init_config/n{}/'.format(nparticle)

    # filename is list : filename[0] : train; filename[1] : valid; filename[2] : test
    filename     = sorted(glob.glob( root_path + 'run*/' + 'nparticle{}_new_nsim_rho0.1_T{}'.format(nparticle, temp) + '_pos_*_sampled.pt'), key=keyFunc)
    print(filename)

    phase_space = phase_space()

    hamiltonian_obj = MD_parameters.hamiltonian_obj

    linear_integrator_obj = linear_integrator(linear_velocity_verlet)

    terms = hamiltonian_obj.get_terms()
    print(terms)

    # nsamples from mc step generated for training data
    q_list_train, p_list_train = torch.load(filename[0])

    phase_space.set_q(q_list_train)
    phase_space.set_p(p_list_train)
    u_term1 = terms[0].energy(phase_space)

    # nsamples from mc step generated for valid data
    q_list_valid, p_list_valid = torch.load(filename[1])

    phase_space.set_q(q_list_valid)
    phase_space.set_p(p_list_valid)
    u_term2 = terms[0].energy(phase_space)

    plt.title('T={}'.format(temp),fontsize=15)
    plt.plot(u_term1,'k-', label = 'train: each {} mcs from {} random samplings'.format(mcstep, new_mcs))
    plt.plot(u_term2,'r-', label = 'valid: each {} mcs from {} random samplings'.format(int(mcstep/10), int(new_mcs/10)))
    plt.xlabel('mcs',fontsize=20)
    plt.ylabel(r'$U_{ij}$',fontsize=20)
    plt.legend()
    plt.show()

    # plt.xlim(xmin=-5.1, xmax = -4.3)
    fig, ax = plt.subplots()
    plt.hist(u_term1.numpy(), bins=100, color='k', alpha = 0.5, label = 'train: histogram of {} samples'.format(len(q_list_train)))
    plt.hist(u_term2.numpy(), bins=100, color='r', alpha = 0.5, label = 'valid: histogram of {} samples'.format(len(q_list_valid)))
    plt.xlabel(r'$U_{ij}$',fontsize=20)
    plt.ylabel('hist', fontsize=20)
    anchored_text = AnchoredText('nparticle={} boxsize={:.3f} temp={} data split = train {} / valid {}'.format(nparticle, boxsize, temp, len(q_list_train), len(q_list_valid)), loc= 'upper left', prop=dict(fontweight="normal", size=12))
    ax.add_artist(anchored_text)

    plt.legend()
    plt.show()

