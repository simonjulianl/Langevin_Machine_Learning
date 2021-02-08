import math

class MC_parameters:

    seed = 23645
    kB = 1.0
    temperature = 0.04
    nsamples = 10
    nparticle = 4
    DIM = 2
    mass = 1
    rho = 0.1
    boxsize = math.sqrt(nparticle / rho)
    iterations = 1200
    DISCARD = 200
    num_interval = iterations - DISCARD
    used_iterations = nsamples
    interval = num_interval // used_iterations # take mc step every interval
    dq = 0.015
