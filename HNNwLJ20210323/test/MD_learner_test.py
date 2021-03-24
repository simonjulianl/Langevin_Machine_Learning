import sys
import os
sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("../parameters"))

from phase_space.phase_space import phase_space
from integrator import linear_integrator
from parameters.MC_parameters import MC_parameters
from parameters.MD_parameters import MD_parameters
# from HNN import pair_wise_HNN
from HNN import field_HNN
from HNN.MD_learner import MD_learner
from HNN.models import fields_unet
# from HNN.models import pair_wise_MLP
# from HNNwLJ20210128.HNN.models import pair_wise_zero

import torch

if __name__ == '__main__':

    nsamples = MD_parameters.nsamples
    nparticle = MC_parameters.nparticle
    npixels = MD_parameters.npixels

    seed = 9372211
    torch.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True  # Processing speed may be lower then when the models functions nondeterministically.
    # torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)

    phase_space = phase_space()

    linear_integrator_obj = linear_integrator(MD_parameters.integrator_method)
    # pair_wise_HNN_obj = pair_wise_HNN(pair_wise_MLP())
    field_HNN_obj = field_HNN(fields_unet(2, 32, 2), linear_integrator_obj)

    print(field_HNN_obj)
    #
    # # q_list = [[[-1.5, 0],[1, 1]]]
    # # q_list = [[[1,0],[0.2,0.21]]]  # particles too close
    # # q_list = [[[-2.,-2.],[1.,1.]]]
    # q_list = [[[-1, -1],[1, 1]]]
    # p_list = [[[0, 0],[0, 0]]]
    #
    # # q_list = [[[-0.62068786, - 0.77235929],[1.23484839, - 1.33486261],[0.12320894, - 1.58505487],[0.42893553, - 0.5222273]]]
    # # p_list = [[[0, 0],[0, 0],[0, 0],[0, 0]]]
    #
    # _q_list_in, _p_list_in = torch.tensor([q_list, p_list], dtype=torch.float64)
    #
    # phase_space.set_q(_q_list_in)
    # phase_space.set_p(_p_list_in)
    #
    # field_HNN_obj.phi_field4cnn(phase_space)
    # field_HNN_obj.p_field4cnn()
    #
    # tau = torch.tensor([0.1])
    # tau = torch.unsqueeze(tau, dim=0)
    #
    # field_HNN_obj.fields2cnn(phase_space, tau)
    # field_HNN_obj.force4nparticle(phase_space, 4) # nsamples x (fx, fy) x npixels x npixels


    # noMLhamiltonian = super(pair_wise_HNN, pair_wise_HNN_obj)

    uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])
    base_dir = uppath(__file__, 2)
    init_path = base_dir + '/init_config/'
    print(init_path)
    torch.autograd.set_detect_anomaly(True) # get the method causing the NANs

    MD_learner = MD_learner(linear_integrator_obj, field_HNN_obj, phase_space, init_path)
    # MD_learner.train_valid_epoch()
    # pred = MD_learner.pred_qnp(filename ='./init_config/N_particle{}_samples{}_rho0.1_T0.04_pos_sampled.pt'.format(nparticle, nsamples))