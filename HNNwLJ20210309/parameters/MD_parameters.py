from integrator import methods
import math

class MD_parameters:

    seed = 4662570 # index for preparing data
    nsamples = 6 # total train/valid 20900
    nsamples_batch = 2
    pixels_batch = 32
    nsamples_ML = 2
    epsilon = 1.
    sigma = 1.
    mass = 1
    temp_list = [0.04]
    npixels = 32
    tau_short = 0.01  # short time step for label
    tau_long = 0.1
    max_ts = 0.4
    iteration_batch = 2
    #iteration_batch = int(max_ts / tau_short) // int(tau_long / tau_short)
    integrator_method = methods.linear_velocity_verlet