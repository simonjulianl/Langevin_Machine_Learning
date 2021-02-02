import math

class LJ_parameters:

    nparticle = 2
    epsilon = 1.
    sigma = 1.
    rho = 0.1
    boxsize = math.sqrt(nparticle / rho)
    # boxsize = 6.