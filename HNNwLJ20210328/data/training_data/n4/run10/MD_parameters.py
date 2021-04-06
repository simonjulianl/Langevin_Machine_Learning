from integrator.methods           import linear_velocity_verlet
from hamiltonian.noML_hamiltonian import noML_hamiltonian
#from HNN.models.pairwise_MLP      import pairwise_MLP
#from HNN.pairwise_HNN             import pairwise_HNN

class MD_parameters:

    mode               = 'train'     # set mode train or valid or test for filename

    init_qp_path       = '../data/init_config/n2/run10/'
    training_path      = '../data/training_data/n2/run10/'
    init_qp_filename   = init_qp_path + 'nparticle2_new_nsim_rho0.1_T0.04_pos_{}_sampled.pt'.format(mode)
    training_filenames = training_path + 'nparticle2_new_nsim_rho0.1_T0.04_pos_{}_sampled'.format(mode)

    nsamples = 2        # train 19000 = T0.04: 5000, T0.16: 6000, T0.32: 8000
                        # valid 1900  = T0.04: 500 , T0.16: 600 , T0.32: 800
                        # test 1000 when noML

    tau_short = 0.001                                   # short time step for label
    append_strike = 20                                  # number of short steps to make one long step
    niter_tau_long  = 10                                # number of MD steps for long tau
    save2file_strike = 20                               # number of short steps to save to file ; save2file_strike >= append_strike
    tau_long = append_strike * tau_short                # value of tau_long
    niter_tau_short = niter_tau_long * append_strike    # number of MD steps for short tau

    integrator_method = linear_velocity_verlet.linear_velocity_verlet
    integrator_method_backward = linear_velocity_verlet.linear_velocity_verlet_backward

    hamiltonian_obj = noML_hamiltonian()
    #hamiltonian_obj = pairwise_HNN(pairwise_MLP())