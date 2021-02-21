import sys
import os
sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("../parameters"))

from HNN.pair_wise_HNN import pair_wise_HNN
from phase_space.phase_space import phase_space
from integrator import linear_integrator
from parameters.MD_parameters import MD_parameters
from HNN.models import pair_wise_MLP
import torch

if __name__ == '__main__':

    # nsamples = MD_parameters.nsamples
    mass = MD_parameters.mass
    boxsize = MD_parameters.boxsize
    # nparticle = MD_parameters.nparticle
    # DIM = MD_parameters.DIM

    phase_space = phase_space()

    pair_wise_HNN_obj = pair_wise_HNN(pair_wise_MLP())
    linear_integrator_obj = linear_integrator(MD_parameters.integrator_method)

    # filename = '../init_config/nparticle{}_new_nsim_rho0.1_T0.04_pos_valid_sampled.pt'.format(nparticle)
    # q_list, p_list = torch.load(filename)

    # q_list = [[[3,2],[2.2,1.21]]]
    # p_list = [[[0.1,0.1],[0.1,0.1]]]
    q_list = [[[-0.62068786, - 0.77235929],[1.23484839, - 1.33486261],[0.12320894, - 1.58505487],[0.42893553, - 0.5222273]]]
    p_list = [[[0,0],[0,0],[0,0],[0,0]]]

    q_list, p_list = torch.tensor([q_list,p_list])

    noMLhamiltonian = super(type(pair_wise_HNN_obj), pair_wise_HNN_obj)

    # print(noMLhamiltonian.hi())
    terms = noMLhamiltonian.get_terms()

    phase_space.set_q(q_list)

    nsamples, nparticle, DIM = q_list.shape

    for z in range(nsamples):

        _, d = phase_space.paired_distance_reduced(q_list[z], nparticle, DIM)


