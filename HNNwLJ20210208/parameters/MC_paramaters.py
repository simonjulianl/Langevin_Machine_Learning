import math

class MC_parameters:

    seed = 23645
    kB = 1.0
    temperature = 0.04
    new_mcs = 120
    nsamples = 1
    nparticle = 4
    DIM = 2
    mass = 1
    rho = 0.1
    boxsize = math.sqrt(nparticle / rho)
    DISCARD = 4000
    iterations = DISCARD + 1  #  1200
    num_interval = iterations - DISCARD
    used_iterations = nsamples
    interval = num_interval // used_iterations # take mc step every interval
    dq = 0.015
