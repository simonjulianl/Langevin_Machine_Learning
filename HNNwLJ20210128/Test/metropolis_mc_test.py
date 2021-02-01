import torch
import math
from HNNwLJ20210128.integrator.metropolis_mc import metropolis_mc
from HNNwLJ20210128.phase_space.phase_space import phase_space
from HNNwLJ20210128.HNN.pair_wise_HNN import pair_wise_HNN
from HNNwLJ20210128.HNN.models.pair_wise_MLP import pair_wise_MLP
from HNNwLJ20210128.parameters.MC_paramaters import MC_parameters


if __name__ == '__main__':

    mass = MC_parameters.mass


    phase_space = phase_space()
    pair_wise_HNN_obj = pair_wise_HNN(pair_wise_MLP())

    noML_hamiltonian = super(type(pair_wise_HNN_obj), pair_wise_HNN_obj)

    mc = metropolis_mc( noML_hamiltonian, phase_space)
    mc.integrate()