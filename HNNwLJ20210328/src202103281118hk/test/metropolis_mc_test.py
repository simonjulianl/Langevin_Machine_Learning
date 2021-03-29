import sys
import os
sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("../parameters"))

from integrator.metropolis_mc import metropolis_mc
from phase_space.phase_space import phase_space
from HNN.pair_wise_HNN import pair_wise_HNN
from HNN.models.pair_wise_MLP import pair_wise_MLP
from parameters.MC_parameters import MC_parameters


if __name__ == '__main__':

    import numpy as np
    import random

    np.random.seed(0)
    random.seed(43893324)

    mass = MC_parameters.mass


    phase_space = phase_space()
    pair_wise_HNN_obj = pair_wise_HNN(pair_wise_MLP())

    noML_hamiltonian = super(type(pair_wise_HNN_obj), pair_wise_HNN_obj)

    mc = metropolis_mc()
    _, U, _, _ = mc.step(noML_hamiltonian, phase_space)

    print(U)