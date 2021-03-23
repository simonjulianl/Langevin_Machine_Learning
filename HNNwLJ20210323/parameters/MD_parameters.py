from integrator import methods
import math

class MD_parameters:

    seed = 4662570 # index for preparing data
    nsamples = 6 # total train/valid 20900 / for predict 20900 / for inegrate 1000
    nsamples_batch = 10
    pixels_batch = 32
    nsamples_ML = 1
    epsilon = 1.
    sigma = 1.
    mass = 1
    temp_list = [0.04]
    npixels = 32
    # tau_short = 0.001  # short time step for label
    # tau_long = 0.1
    # tau_short_list = [0.002, 0.001, 0.0008, 0.0006, 0.0004]
    # tau_long_list = [0.2, 0.1, 0.08, 0.06, 0.04]
    tau_short_list = [0.001]
    tau_long_list = [0.1]
    tau_pair = int(tau_long_list[0] / tau_short_list[0])
    max_ts = 10.
    # iteration_batch = 10
    iteration_batch = 1   #  :train/valid,  int(max_ts ) : test
    ML_iteration_batch = 2
    crash_duplicate_ratio = 0.4
    integrator_method = methods.linear_velocity_verlet