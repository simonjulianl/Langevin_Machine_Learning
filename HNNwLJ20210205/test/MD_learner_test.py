import sys
import os
sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("../parameters"))

from phase_space.phase_space import phase_space
from integrator import linear_integrator
from parameters.MD_paramaters import MD_parameters
from HNN import pair_wise_HNN
from HNN.MD_learner import MD_learner
from HNN.models import pair_wise_MLP
# from HNNwLJ20210128.HNN.models import pair_wise_zero

import torch

if __name__ == '__main__':

    nsamples = MD_parameters.nsamples
    nparticle = MD_parameters.nparticle

    seed = 9372211
    torch.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True  # Processing speed may be lower then when the models functions nondeterministically.
    # torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)

    phase_space = phase_space()

    linear_integrator_obj = linear_integrator(MD_parameters.integrator_method)
    pair_wise_HNN_obj = pair_wise_HNN(pair_wise_MLP())

    print(pair_wise_HNN_obj)

    # noMLhamiltonian = super(pair_wise_HNN, pair_wise_HNN_obj)

    filename1 = '../init_config/nparticle{}_new_nsim_rho0.1_T0.04_pos_train_sampled.pt'.format(nparticle)
    filename2 = '../init_config/nparticle{}_new_nsim_rho0.1_T0.04_pos_valid_sampled.pt'.format(nparticle)

    torch.autograd.set_detect_anomaly(True) # get the method causing the NANs

    MD_learner = MD_learner(linear_integrator_obj, pair_wise_HNN_obj, phase_space, filename1, filename2)
    MD_learner.train_valid_epoch()
    # pred = MD_learner.pred_qnp(filename ='./init_config/N_particle{}_samples{}_rho0.1_T0.04_pos_sampled.pt'.format(nparticle, nsamples))
