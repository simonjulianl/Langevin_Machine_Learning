import math

class MC_parameters:

    seed = 2364532  # 23645 for train / 35029 for valid
    kB = 1.0
    temperature = 0.04
    new_mcs = 4
    nsamples = 5
    nparticle = 4
    DIM = 2
    mass = 1
    rho = 0.1
    boxsize = math.sqrt(nparticle / rho)
    DISCARD = 40
    interval = 40 # take mc step every interval
    num_interval = interval * nsamples
    iterations = num_interval + DISCARD  #  1200
    dq = 0.015 #T0.04 -> 0.015, T0.16 -> 0.035, T0.32 -> 0.08

