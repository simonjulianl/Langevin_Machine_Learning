from integrator import methods
import math

class MD_parameters:

    data_path = '/training_data/n2/run1'
    nsamples = 2 # total train/valid 20900 / when predict, set 20900 to load file / set 1000 when noML
    nsamples_batch = 2  # num. of batch for nsamples when prepare data before ML
    nsamples_ML = 1
    epsilon = 1.
    sigma = 1.
    mass = 1
    temp_list = [0.32]

    tau_short = 0.001                                    # short time step for label
    append_strike = 20                                   # number of short steps to make one long step
    niter_tau_long  = 1                                  # number of MD steps for long tau
    save2file_strike = 20                               # number of short steps to save to file
    tau_long = append_strike * tau_short                # value of tau_long
    niter_tau_short = niter_tau_long * append_strike    # number of MD steps for short tau


    #nstack = 1  # 100 th step at short time step paired with first large time step
    #iteration_batch = 1   #  use linear_integrator; setting 1 : train/valid,  int(max_ts ) : test or more iteration for gold standard

    crash_duplicate_ratio = 0.4 # use data_io
    integrator_method = methods.linear_velocity_verlet

    # optical flow
    pixels_batch = 32
    npixels = 32
    ML_iteration_batch = 2 #  set 50 the num. of saved files when test  1000 /
