import math

class MC_parameters:

    nparticle = 2
    temperature = 0.32
    mode = 'train'                              # set mode train or valid or test for filename

    init_path           = '../data/init_config/n{}/run11/'.format(nparticle)
    filename            = init_path + 'nparticle{}_new_nsim_rho0.1_T{}_pos_{}_sampled.pt'.format(nparticle, temperature, mode)

    seed = 89236                                # set different seed for generate data for train/valid/test
                                                # nsamples 20900 ->  23645 for train / 35029 for valid
                                                # nsamples 41800 ->  89236 for train / 49832 for valid
                                                # nsamples 1000  -> 15343 for test
    DIM = 2
    rho = 0.1                                   # density
    boxsize = math.sqrt(nparticle / rho)        # boxsize = sqrt(nparticle / density)

    new_mcs = 2                                 # the number of samples for mc
    mcstep = 10                                 # mc step each sample
    max_energy = 1e3                            # energy threshold
    DISCARD = 50                                # discard initial mc steps
    interval = 4                                # take mc step every given interval
    iterations = (interval * mcstep) + DISCARD  # iterations included discard
    dq = 0.015                                  # displacement to increase acceptance rate

    # 4 particles : T0.04 -> 0.015, T0.16 -> 0.035, T0.32 -> 0.08  DISCARD 4000  interval 40
    # 8 particles :       -> 0.01         -> 0.023,       -> 0.04  DISCARD 8000  interval 40
    # 16 particles :      -> 0.006        -> 0.015        -> 0.02  DISCARD 16000 interval 40


