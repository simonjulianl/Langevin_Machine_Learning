from HNNwLJ20210128.phase_space.phase_space import phase_space
from HNNwLJ20210128.integrator import linear_integrator
from HNNwLJ20210128.parameters.MD_paramaters import MD_parameters
from HNNwLJ20210128.HNN import pair_wise_HNN
from HNNwLJ20210128.HNN.MD_learner import MD_learner
from HNNwLJ20210128.HNN.models import pair_wise_MLP
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

    filename = '../init_config/N_particle{}_samples{}_rho0.1_T0.04_pos_sampled.pt'.format(nparticle, nsamples)

    torch.autograd.set_detect_anomaly(True) # get the method causing the NANs

    MD_learner = MD_learner(linear_integrator_obj, pair_wise_HNN_obj, phase_space, filename)
    MD_learner.train_valid_epoch()
    # pred = MD_learner.pred_qnp(filename ='./init_config/N_particle{}_samples{}_rho0.1_T0.04_pos_sampled.pt'.format(nparticle, nsamples))
