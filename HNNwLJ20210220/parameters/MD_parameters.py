from integrator import methods
import math

class MD_parameters:

    seed = 4662570 # index for preparing data
    gen_nsamples = 30
    nsamples = 30
    nsamples_ML = 1
    epsilon = 1.
    sigma = 1.
    mass = 1
    temp_list = [0.04]
    npixels = 32
    tau_short = 0.001  # short time step for label
    tau_long = 0.1
    integrator_method = methods.linear_velocity_verlet