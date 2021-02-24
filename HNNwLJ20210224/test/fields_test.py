import sys
import os
sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("../parameters"))

import numpy as np
from HNN.field_HNN import field_HNN
from fields.phi_fields import phi_fields
from fields.momentum_fields import momentum_fields
from HNN.models import pair_wise_MLP
from phase_space.phase_space import phase_space
from parameters.MD_parameters import MD_parameters
from integrator import linear_integrator
import matplotlib.pyplot as plt
import torch

if __name__ == '__main__':

    nsamples = MD_parameters.nsamples
    mass = MD_parameters.mass
    boxsize = MD_parameters.boxsize
    nparticle = MD_parameters.nparticle
    DIM = MD_parameters.DIM
    npixels = MD_parameters.npixels

    print('nparticle', nparticle, 'boxsize', boxsize , 'npixels', npixels)
    phase_space = phase_space()

    linear_integrator_obj = linear_integrator(MD_parameters.integrator_method)
    pair_wise_HNN_obj = field_HNN(pair_wise_MLP(),linear_integrator_obj)
    noMLhamiltonian = super(type(pair_wise_HNN_obj), pair_wise_HNN_obj)

    # load filename
    # filename = '../init_config/N_particle{}_samples{}_rho0.1_T0.04_pos_sampled.pt'.format(nparticle, nsamples_label)
    # q_list = [[[-1, -1],[1, 1]]]
    q_list = [[[-2.2, 0],[2.2, 0]]]
    # q_list = [[[-1.7, 0],[2.2, 0]]]
    # q_list = [[[1., 0],[0.2, 0.21]]]
    p_list = [[[0, 0],[0, 0]]]
    # q_list = [[[-0.62068786, - 0.77235929],[1.23484839, - 1.33486261],[0.12320894, - 1.58505487],[0.42893553, - 0.5222273]]]
    # p_list = [[[0,0],[0,0],[0,0],[0,0]]]

    _q_list_in, _p_list_in = torch.tensor([q_list, p_list],dtype=torch.float64)

    phi_fields_obj = phi_fields(npixels, noMLhamiltonian)

    phase_space.set_q(_q_list_in)
    phase_space.set_p(_p_list_in)

    pair_wise_HNN_obj.phi_field4cnn(phase_space)
    pair_wise_HNN_obj.p_field4cnn()