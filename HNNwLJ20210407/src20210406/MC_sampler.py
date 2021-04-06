from hamiltonian.noML_hamiltonian import noML_hamiltonian
from phase_space import phase_space
from utils.data_io import data_io
from parameters.MC_parameters import MC_parameters
from integrator.metropolis_mc import metropolis_mc
from integrator.momentum_sampler import momentum_sampler

import sys
import torch

import numpy as np
import random

if __name__=='__main__':
    # run something like this
    # python MC_sampler.py ../data/init_config/n2run@@/MC_config.dict

    argv = sys.argv
    MCjson_file = argv[1]
    MC_parameters.load_dict(MCjson_file)

    seed        = MC_parameters.seed        # set different seed for generate data (train, valid, and test)
    interval    = MC_parameters.interval    # take mc steps every given interval

    # io varaiables
    filename    = MC_parameters.filename

    # seed setting
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    data_io_obj     = data_io()
    phase_space     = phase_space.phase_space()
    noMLhamiltonian = noML_hamiltonian()

    metropolis_mc   = metropolis_mc()

    q_hist, U, ACCRatio, spec = metropolis_mc.step(noMLhamiltonian, phase_space)

    # take q every interval
    q_hist = q_hist[:, 0::interval, :, :]
    # shape is  [mc nsamples, mc interval step, nparticle, DIM]
    q_hist = torch.reshape(q_hist, (-1, q_hist.shape[2], q_hist.shape[3]))
    # shape is  [(mc nsamples x mc step), nparticle, DIM]

    # momentum sampler
    momentum_sampler = momentum_sampler(q_hist.shape[0])
    p_hist = momentum_sampler.momentum_samples()
    # shape is  [nsamples, nparticle, DIM]   ; nsamples = mc nsamples x mc step

    qp_list = torch.stack((q_hist,p_hist),dim=1)
    # shape is [nsamples, (q, p), nparticle, DIM]

    qp_list = torch.unsqueeze(qp_list,dim=2)
    # shape is [nsamples, (q, p), 1, nparticle, DIM]

    data_io_obj.write_trajectory_qp(filename, qp_list)
    print('file write dir:', filename)


