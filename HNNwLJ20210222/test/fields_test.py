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

    phase_space = phase_space()

    pair_wise_HNN_obj = field_HNN(pair_wise_MLP())
    linear_integrator_obj = linear_integrator(MD_parameters.integrator_method)
    noMLhamiltonian = super(type(pair_wise_HNN_obj), pair_wise_HNN_obj)

    def show_grid_nparticles(q_list, _grid_list):

        plt.plot(_grid_list[:, 0], _grid_list[:, 1], marker='.', color='k', linestyle='none', markersize=12)
        plt.plot(q_list[:, :, 0], q_list[:, :, 1], marker='x', color='r', linestyle='none', markersize=12)
        plt.show()
        plt.close()

    # load filename
    # filename = '../init_config/N_particle{}_samples{}_rho0.1_T0.04_pos_sampled.pt'.format(nparticle, nsamples_label)
    q_list = [[[-1, -1],[1, 1]]]
    p_list = [[[0, 0],[0, 0]]]

    _q_list_in, _p_list_in = torch.tensor([q_list, p_list],dtype=torch.float64)

    phi_fields_obj = phi_fields(npixels, noMLhamiltonian)
    # show_grid_nparticles(_q_list_in, _grid_list)

    # _phi_field_in = phi_fields_obj.show_grid_nparticles(_q_list_in, 'hi')

    phase_space.set_q(_q_list_in)
    # phase_space.set_p(_p_list_in)

    phi_field_in = phi_fields_obj.phi_field(phase_space)
    phi_fields_obj.show_gridimg()

    nsamples_cur = MD_parameters.nsamples
    tau_cur = MD_parameters.tau_short
    MD_iterations = int(MD_parameters.tau_short / MD_parameters.tau_short)

    phase_space.set_q(_q_list_in)
    phase_space.set_p(_p_list_in)

    _q_list_nx, _p_list_nx = linear_integrator_obj.step(noMLhamiltonian, phase_space, MD_iterations, nsamples_cur, tau_cur)
    _q_list_nx = _q_list_nx[-1].type(torch.float64)  # only take the last from the list
    _p_list_nx = _p_list_nx[-1].type(torch.float64)

    phase_space.set_q(_q_list_nx)
    # phase_space.set_p(_p_list_nx)
    phi_field_nx = phi_fields_obj.phi_field(phase_space)
    phi_fields_obj.show_gridimg()

    print('init q', _q_list_in, 'next q', _q_list_nx)

    momentum_fields_obj = momentum_fields(phi_field_in, phi_field_nx)
    flow_vectors = momentum_fields_obj.p_field()
    momentum_fields_obj.visualize_flow_file(flow_vectors)