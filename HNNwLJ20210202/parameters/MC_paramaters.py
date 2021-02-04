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
    interval = 4 # take mc step every interval
    iterations = 100
    DISCARD = iterations - (nsamples * interval)
    dq = 0.1