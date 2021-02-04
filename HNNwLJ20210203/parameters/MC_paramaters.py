import math

class MC_parameters:

    seed = 23645
    kB = 1.0
    temperature = 0.04
    nsamples = 20
    nparticle = 2
    DIM = 2
    mass = 1
    rho = 0.1
    boxsize = math.sqrt(nparticle / rho)
    iterations = 602000
    DISCARD = 2000
    num_interval = iterations - DISCARD
    used_iterations = 20000
    interval = num_interval //  used_iterations # take mc step every interval
    dq = 0.1
