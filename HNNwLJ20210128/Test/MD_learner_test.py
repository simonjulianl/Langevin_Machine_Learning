from HNNwLJ20210128.hamiltonian import hamiltonian
from HNNwLJ20210128.hamiltonian.kinetic_energy import kinetic_energy
from HNNwLJ20210128.hamiltonian.lennard_jones import lennard_jones
from HNNwLJ20210128.hamiltonian.LJ_term import LJ_term
from HNNwLJ20210128.phase_space.phase_space import phase_space
from HNNwLJ20210128.integrator import linear_integrator
from HNNwLJ20210128.integrator.methods import linear_velocity_verlet
from HNNwLJ20210128.parameters.MD_paramaters import MD_parameters
from HNNwLJ20210128.parameters.ML_paramaters import ML_parameters
from HNNwLJ20210128.pair_wise_HNN.data_io import data_io
from HNNwLJ20210128.pair_wise_HNN import pair_wise_HNN
from HNNwLJ20210128.pair_wise_HNN.MD_learner import MD_learner
from HNNwLJ20210128.pair_wise_HNN.models import pair_wise_MLP
from HNNwLJ20210128.pair_wise_HNN.loss import qp_MSE_loss
import torch
import math

import torch.optim as optim

if __name__ == '__main__':

    nsamples_cur = MD_parameters.nsamples_cur
    nsamples_label = MD_parameters.nsamples_label
    nsamples_ML = MD_parameters.nsamples_ML
    mass = MD_parameters.mass
    epsilon = MD_parameters.epsilon
    sigma = MD_parameters.sigma
    boxsize = MD_parameters.boxsize
    nparticle = MD_parameters.nparticle
    DIM = MD_parameters.DIM
    MD_iterations = MD_parameters.MD_iterations # 10 steps to pair with large time step = 0.1
    tau_cur = MD_parameters.tau_cur
    tau_long = MD_parameters.tau_long # short time step for label
    tau_short = MD_parameters.tau_short
    nepoch = ML_parameters.nepoch
    lr = ML_parameters.lr
    MLP_input = ML_parameters.MLP_input
    MLP_nhidden = ML_parameters.MLP_nhidden


    seed = 9372211
    torch.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True  # Processing speed may be lower then when the models functions nondeterministically.
    # torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)

    # noML_hamiltonian
    noML_hamiltonian = hamiltonian.hamiltonian()
    lj_term = LJ_term(epsilon=epsilon,sigma=sigma,boxsize=boxsize)
    noML_hamiltonian.append(lennard_jones(lj_term, boxsize))
    noML_hamiltonian.append(kinetic_energy(mass))

    phase_space = phase_space()

    # state = {
    #     'nsamples_cur': 0,
    #     'nsamples_label': nsamples_label,
    #     'nsamples_ML': nsamples_ML,
    #     'nparticle': nparticle,
    #     'DIM': DIM,
    #     'boxsize': boxsize,
    #     'MD_iterations': 0,
    #     'tau_cur': 0.0,
    #     'tau_long': tau_long,  # for MD
    #     'tau_short': tau_short,  # for label (gold standard)
    #     'integrator_method': linear_velocity_verlet,
    #     'phase_space': phase_space,
    #     'pb_q': pb,
    #     'nepochs': nepochs,
    #     'n_hidden': n_hidden
    # }

    MLP = pair_wise_MLP(MLP_input, MLP_nhidden)
    # MLP = models.pair_wise_zero(n_input, n_hidden)
    opt = optim.Adam(MLP.parameters(), lr=lr)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    setting = {
        'opt': opt,
        'loss': qp_MSE_loss,
        'MLP': MLP,
        '_device': device
    }

    state.update(setting)
    #=== prepare label ====================================================#
    filename = '../init_config/N_particle{}_samples{}_rho0.1_T0.04_pos_sampled.pt'.format(nparticle, nsamples_label)
    # dataset = dataset_split(filename, **state)
    # train_data, valid_data = dataset.hamiltonian_dataset(ratio=0.8)
    # # train_label = dataset.phase_space2label(train_data, linear_integrator, noML_hamiltonian)
    # valid_q_label, valid_p_label = dataset.phase_space2label(valid_data, linear_integrator, noML_hamiltonian)
    #===== end ============================================================#

    torch.autograd.set_detect_anomaly(True) # get the method causing the NANs

    MD_learner = MD_learner(linear_integrator, noML_hamiltonian, pair_wise_HNN,
                                          filename, **state)

    MD_learner.train_valid_epoch()
    # pred = MD_learner.pred_qnp(filename ='./init_config/N_particle{}_samples{}_rho0.1_T0.04_pos_sampled.pt'.format(nparticle, nsamples_label))
