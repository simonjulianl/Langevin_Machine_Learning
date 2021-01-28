import math

class MD_parameters:

    nsamples_cur = 0
    nsamples_label = 1
    nsamples_ML = 1
    nparticle = 2
    DIM = 2
    mass = 1
    epsilon = 1.
    sigma = 1.
    rho = 0.1
    # boxsize = math.sqrt(nparticle / rho)
    boxsize = 6
    MD_iterations = 0 # 10 steps to pair with large time step = 0.1
    tau_cur = 0.0
    tau_short = 0.01  # short time step for label
    tau_long = 0.1