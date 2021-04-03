from integrator.methods import linear_velocity_verlet
from hamiltonian.noML_hamiltonian import noML_hamiltonian
# from HNN.pair_wise_HNN import pair_wise_HNN
# from HNN.models.pair_wise_MLP import pair_wise_MLP

class MD_parameters:

    mode               = 'train'     # set mode train or valid or test for filename

    init_qp_path       = '../data/init_config/n4/'
    data_path          = '../data/training_data/n2/run12/'
    init_qp_filename   = init_qp_path + 'nparticle4_new_nsim_rho0.1_allT_{}_sampled.pt'.format(mode)
    data_filenames     = data_path    + 'nparticle4_new_nsim_rho0.1_allT_{}_sampled'.format(mode)

    #nsamples = 8        # train 19000 = T0.04: 5000, T0.16: 6000, T0.32: 8000
                        # valid 1900  = T0.04: 500 , T0.16: 600 , T0.32: 800
                        # test 1000 when noML

    tau_short = 0.1                                    # short time step for label
    append_strike = 20                                   # number of short steps to make one long step
    niter_tau_long  = 10                                 # number of MD steps for long tau
    save2file_strike = 200                               # number of short steps to save to file; save2file_strike >= append_strike
    tau_long = append_strike * tau_short                # value of tau_long
    niter_tau_short = niter_tau_long * append_strike    # number of MD steps for short tau

    integrator_method = linear_velocity_verlet.linear_velocity_verlet
    integrator_method_backward = linear_velocity_verlet.linear_velocity_verlet_backward

    hamiltonian_obj = noML_hamiltonian()
    # hamiltonian_obj = pair_wise_HNN(pair_wise_MLP()) <- import error(sequential), cannot use this
