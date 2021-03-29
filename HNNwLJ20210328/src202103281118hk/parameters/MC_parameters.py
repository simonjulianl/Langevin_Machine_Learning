import math

class MC_parameters:

    # nsamples 20900 ->  23645 for train / 35029 for valid
    # nsamples 41800 ->  89236 for train / 49832 for valid
    # nsamples 1000 -> 15343 for test
    seed = 89236
    mode = 'test' # for test set mode 'test'
    kB = 1.0
    temperature = 0.32
    new_mcs = 2
    mcstep = 4
    nparticle = 2
    DIM = 2
    mass = 1
    rho = 0.1
    boxsize = math.sqrt(nparticle / rho)
    max_energy = 1e3
    DISCARD = 50
    interval = 4 # take mc step every interval
    num_interval = interval * mcstep
    iterations = num_interval + DISCARD  #  1200
    # 4 particles : T0.04 -> 0.015, T0.16 -> 0.035, T0.32 -> 0.08  DISCARD 4000 interval 40
    # 8 particles :       -> 0.01         -> 0.023,       -> 0.04  DISCARD 4000 interval 40
    # 16 particles :      -> 0.006        -> 0.015        -> 0.02  DISCARD 4000 interval 40
    dq = 0.015


