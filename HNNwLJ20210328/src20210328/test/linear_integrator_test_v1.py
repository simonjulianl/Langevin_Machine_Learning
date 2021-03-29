from HNNwLJ20210128.phase_space.phase_space import phase_space
from HNNwLJ20210128.integrator import linear_integrator
from HNNwLJ20210128.HNN.pair_wise_HNN import pair_wise_HNN
from HNNwLJ20210128.parameters.MD_paramaters import MD_parameters
from HNNwLJ20210128.parameters.ML_paramaters import ML_parameters
from HNNwLJ20210128.HNN.models import pair_wise_MLP

import matplotlib.pyplot as plt
import torch

if __name__ == '__main__':


    nsamples = 1
    select_nsamples = 1
    nsamples_ML = 1
    nparticle = 2
    DIM = 2
    # MD_iterations = MD_parameters.MD_iterations # 10 steps to pair with large time step = 0.1
    tau_long = 0.1 # short time step for label
    tau_short = 0.0001

    # q_list = [[[2,1],[5,4],[1,3]]]
    # p_list = [[[0.1,0.1],[0.2,0.2],[0.2,0.2]]]
    # q_list_ = [[[3,2],[2.2,1.21]]]
    # p_list_ = [[[0.,0.],[0.,0.]]]
    q_list_ = [[[-0.8944, -1.8265],[-0.9352, -2.9248],[0.0404, -2.4314],[0.0041,  2.7229]]]
    p_list_ = [[[0.1872,  0.2631],[-0.0559, -0.2942],[0.1030,  0.1788],[0.0288, -0.0531]]]

    q_list_tensor, p_list_tensor = torch.tensor([q_list_,p_list_])

    print('initial ', q_list_tensor, p_list_tensor)

    phase_space = phase_space()
    pair_wise_HNN_obj = pair_wise_HNN(pair_wise_MLP())

    # NoML_hamiltonian.dHdq(phase_space)

    # initial state
    phase_space.set_q(q_list_tensor)
    phase_space.set_p(p_list_tensor)

    # tau = short time step 0.01 and 10 steps to pair with large time step 0.1
    nsamples_cur = 1
    tau_cur = tau_short
    MD_iterations = int(tau_long / tau_cur)

    print('for label : nsamples_cur, tau_cur, MD_iterations')
    print(nsamples_cur, tau_cur, MD_iterations)

    noMLhamiltonian = super(pair_wise_HNN, pair_wise_HNN_obj)
    print('main noMLhamiltonian', noMLhamiltonian)

    q_list_label, p_list_label = linear_integrator(MD_parameters.integrator_method).integrate( noMLhamiltonian, phase_space, MD_iterations, nsamples_cur, tau_cur)

    print('label',q_list_label, p_list_label)

    # initial state
    phase_space.set_q(q_list_tensor)
    phase_space.set_p(p_list_tensor)

    # tau = large time step 0.1 and 1 step
    nsamples_cur = nsamples_ML
    tau_cur = tau_long  # tau = 0.1
    MD_iterations = int(tau_long / tau_cur)

    print('At large time : nsamples_cur, tau_cur, MD_iterations')
    print(nsamples_cur, tau_cur, MD_iterations)
    q_list, p_list = linear_integrator(MD_parameters.integrator_method).integrate( noMLhamiltonian, phase_space, MD_iterations, nsamples_cur, tau_cur)

    print('noML vv',q_list, p_list)
