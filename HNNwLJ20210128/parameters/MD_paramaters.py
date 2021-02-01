from HNNwLJ20210128.integrator import methods
import math

class MD_parameters:

    seed = 42657933
    nsamples = 500
    select_nsamples = 40
    nsamples_ML = 1
    nparticle = 4
    DIM = 2
    mass = 1
    rho = 0.1
    boxsize = math.sqrt(nparticle / rho)
    tau_short = 0.01  # short time step for label
    tau_long = 0.1
    integrator_method = methods.linear_velocity_verlet