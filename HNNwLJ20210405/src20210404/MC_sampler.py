from hamiltonian.noML_hamiltonian import noML_hamiltonian
from phase_space import phase_space
from utils.data_io import data_io
from parameters.MC_parameters import MC_parameters
from integrator.metropolis_mc import metropolis_mc
from integrator.momentum_sampler import momentum_sampler
import torch

import numpy as np
import random

if __name__=='__main__':

    seed        = MC_parameters.seed        # set different seed for generate data (train, valid, and test)
    interval    = MC_parameters.interval    # take mc steps every given interval

    # io varaiables
    filename    = MC_parameters.filename

    # seed setting
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    data_io_obj     = data_io()
    phase_space     = phase_space.phase_space()
    noMLhamiltonian = noML_hamiltonian()

    metropolis_mc   = metropolis_mc()

    q_hist, U, ACCRatio, spec = metropolis_mc.step(noMLhamiltonian, phase_space)

    # take q every interval
    q_hist = q_hist[:, 0::interval] # shape is  [new_mcs, mc interval step, nparticle, DIM]
    q_hist = torch.reshape(q_hist, (-1, q_hist.shape[2], q_hist.shape[3])) # shape is  [(new_mcs x mc step), nparticle, DIM]

    # momentum sampler
    momentum_sampler = momentum_sampler(q_hist.shape[0])
    p_hist = momentum_sampler.momentum_samples()   # shape is  [(new_mcs x mc step), nparticle, DIM]

    qp_list = torch.stack((q_hist,p_hist))         # shape is [(q,p), (new_mcs x mc step), nparticle, DIM]
    qp_list = torch.unsqueeze(qp_list,dim=0)

    data_io_obj.write_init_qp(filename, qp_list)
    print('write dir:', filename)


