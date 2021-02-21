import math

class MC_parameters:

    seed = 23645  # 23645 for train / 35029 for valid
    mode = 'valid'
    kB = 1.0
    temperature = 0.04
    new_mcs = 1
    nsamples = 5
    nparticle = 16
    DIM = 2
    mass = 1
    rho = 0.1
    boxsize = math.sqrt(nparticle / rho)
    DISCARD = 4000
    interval = 40 # take mc step every interval
    num_interval = interval * nsamples
    iterations = num_interval + DISCARD  #  1200
    # 4 particles : T0.04 -> 0.015, T0.16 -> 0.035, T0.32 -> 0.08  DISCARD 4000 interval 40
    # 8 particles :       -> 0.01         -> 0.023,       -> 0.04  DISCARD 4000 interval 40
    # 16 particles :      -> 0.006        -> 0.015        -> 0.02  DISCARD 4000 interval 40
    dq = 0.006


