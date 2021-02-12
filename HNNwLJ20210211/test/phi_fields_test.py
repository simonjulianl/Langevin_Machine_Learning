import sys
import os
sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("../parameters"))

import numpy as np
from HNN.field_HNN import field_HNN
from fields import phi_fields
from HNN.models import pair_wise_MLP
from phase_space.phase_space import phase_space
from parameters.MD_paramaters import MD_parameters
from integrator import linear_integrator
import matplotlib.pyplot as plt
import torch

if __name__ == '__main__':


    gen_nsamples = MD_parameters.gen_nsamples
    nsamples = MD_parameters.nsamples
    mass = MD_parameters.mass
    boxsize = MD_parameters.boxsize
    nparticle = MD_parameters.nparticle
    DIM = MD_parameters.DIM
    npixels = MD_parameters.npixels

    phase_space = phase_space()

    pair_wise_HNN_obj = field_HNN(pair_wise_MLP())
    linear_integrator_obj = linear_integrator(MD_parameters.integrator_method)


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

    phi_fields_obj = phi_fields(npixels)
    # show_grid_nparticles(_q_list_in, _grid_list)

    phase_space.set_q(_q_list_in)
    phase_space.set_p(_p_list_in)

    _phi_field_in = phi_fields_obj.show_grid_nparticles(_q_list_in, 'hi')

    quit()


    phase_space.set_q(_q_list_in)
    phase_space.set_p(_p_list_in)

    print('q,p',phase_space.get_q(),phase_space.get_p())

    state['nsamples_cur'] = state['nsamples_label']  # for one step
    state['tau_cur'] = state['tau_short']  # tau = 0.01
    state['MD_iterations'] = int(state['tau_short'] / state['tau_cur'])  # for one step
    _q_list_nx, _p_list_nx = linear_integrator(**state).integrate(NoML_hamiltonian)

    _q_list_nx = _q_list_nx[-1].type(torch.float64)  # only take the last from the list
    _p_list_nx = _p_list_nx[-1].type(torch.float64)

    phase_space.set_q(_q_list_nx)
    # phase_space.set_p(_p_list_nx)
    print('next q,p', phase_space.get_q(), phase_space.get_p())

    _phi_field_nx = ljbox2gridimg_obj.phi_field(phase_space, pb)
    ljbox2gridimg_obj.show_gridimg()

    optical_flow2img = optical_flow2img(_phi_field_in, _phi_field_nx)
    flow_vectors = optical_flow2img.p_field()
    optical_flow2img.visualize_flow_file(flow_vectors)

