
class MD_config:

    mode               = 'train'     # set mode train or valid or test for filename

    tau_short = 0.001                                   # short time step for label
    append_strike = 100                                 # number of short steps to make one long step
    niter_tau_long  = 1                                 # number of MD steps for long tau
    save2file_strike = 100                              # number of short steps to save to file; save2file_strike >= append_strike
    tau_long = append_strike * tau_short                # value of tau_long
    niter_tau_short = niter_tau_long * append_strike    # number of MD steps for short tau

    init_qp_path       = '../data/init_config/combined/'
    data_path          = '../data/training_data/n2run@@/'
    init_qp_filename   = init_qp_path + 'n2_nsim_rho0.1allT_{}_sampled.pt'.format(mode)
    data_filenames     = data_path    + 'n2_nsim_rho0.1allT_{}_sampled'.format(mode)

