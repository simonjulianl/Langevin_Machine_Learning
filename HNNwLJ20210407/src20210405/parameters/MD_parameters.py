from integrator.methods import linear_velocity_verlet
from hamiltonian.noML_hamiltonian import noML_hamiltonian
from HNN.pairwise_HNN import pairwise_HNN
from HNN.models.pairwise_MLP import pairwise_MLP

class MD_parameters:

    init_qp_path       = '../data/init_config/n2run13/'
    data_path 	       = '../data/test_data/n2run10/'
    init_qp_filename   = init_qp_path + 'n2_nsim_rho0.1T0.04_test_sampled.pt'
    data_filenames = data_path + 'n2_nsim_rho0.1T0.04_test_sampled'

    tau_short = 0.001                                   # short time step for label
    append_strike = 100                                  # number of short steps to make one long step
    niter_tau_long  = 100                                 # number of MD steps for long tau
    save2file_strike = 500                                # number of short steps to save to file
    tau_long = append_strike * tau_short                # value of tau_long
    niter_tau_short = niter_tau_long * append_strike    # number of MD steps for short tau

    integrator_method = linear_velocity_verlet.linear_velocity_verlet
    integrator_method_backward = linear_velocity_verlet.linear_velocity_verlet_backward

    # hamiltonian_obj = noML_hamiltonian()
    hamiltonian_obj = pairwise_HNN(pairwise_MLP(), pairwise_MLP(), tau_long)
