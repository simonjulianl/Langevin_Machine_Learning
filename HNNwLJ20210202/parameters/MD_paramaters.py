from integrator import methods
import math

class MD_parameters:

    seed = 42657
    nsamples = 5
    select_nsamples = 3
    nsamples_ML = 1
    nparticle = 2
    DIM = 2
    mass = 1
    temp = 0.4
    rho = 0.1
    boxsize = math.sqrt(nparticle / rho)
    tau_short = 0.01  # short time step for label
    tau_long = 0.1
    integrator_method = methods.linear_velocity_verlet