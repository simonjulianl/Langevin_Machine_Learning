import math

class MC_parameters:

    seed = 23645
    kB = 1.0
    temperature = 0.04
    nsamples = 5
    nparticle = 2
    DIM = 2
    mass = 1
    rho = 0.1
    boxsize = math.sqrt(nparticle / rho)
    interval = 2 # take mc step every interval
    iterations = 10
    DISCARD = iterations - (nsamples * interval)
    dq = 0.1
