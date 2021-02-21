import sys
import os
sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("../parameters"))

from phase_space.phase_space import phase_space
from integrator import linear_integrator
from HNN.pair_wise_HNN import pair_wise_HNN
from parameters.MD_parameters import MD_parameters
from HNN.models import pair_wise_MLP
from HNN.MD_learner import MD_learner
import torch

if __name__ == '__main__':


    nsamples = MD_parameters.nsamples
    nparticle = MD_parameters.nparticle

    phase_space = phase_space()

    pair_wise_HNN_obj = pair_wise_HNN(pair_wise_MLP())
    linear_integrator_obj = linear_integrator(MD_parameters.integrator_method)

    uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])
    base_dir = uppath(__file__, 2)
    init_path = base_dir + '/init_config/'

    torch.autograd.set_detect_anomaly(True) # get the method causing the NANs

    MD_learner = MD_learner(linear_integrator_obj, pair_wise_HNN_obj, phase_space, init_path)
    print('q shape', MD_learner.valid_label[0].shape)
    print('q list', MD_learner.valid_label[0][:,0])
    print('p shape', MD_learner.valid_label[1].shape)
    print('q list', MD_learner.valid_label[1][:,0])
