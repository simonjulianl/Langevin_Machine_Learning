from integrator import methods
import math

class MD_parameters:

    seed = 4662570 # index for preparing data
    nsamples = 6
    nsamples_batch = 50
    nsamples_ML = 1
    nparticle = 2
    DIM = 2
    epsilon = 1.
    sigma = 1.
    mass = 1
    temp_list = [0.04]
    rho = 0.1
    npixels = 32
    boxsize = math.sqrt(nparticle / rho)
    tau_short = 0.01  # short time step for label
    tau_long = 0.1
    integrator_method = methods.linear_velocity_verlet