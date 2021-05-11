from phase_space.phase_space         import phase_space
from hamiltonian.noML_hamiltonian    import noML_hamiltonian
from utils.data_io                   import data_io

import sys
import torch

if __name__ == '__main__':

    '''python check_gen_by_md.py ../data/gen_by_MD/train/xxx.pt'''

    argv = sys.argv

    infile1   = argv[1]

    phase_space = phase_space()
    hamiltonian_obj = noML_hamiltonian()

    qp_trajectory, _, _, boxsize = data_io.read_trajectory_qp(infile1)
    # init_qp.shape = [nsamples, (q, p), 2, nparticle, DIM]
    print(qp_trajectory.shape)

    init_qp = qp_trajectory[:,:,0,:,:]
    qp_strike_append = qp_trajectory[:,:,1,:,:]
    print('initial qp', init_qp.shape)
    print('qp paired with one large time step', qp_strike_append.shape)

    #terms = hamiltonian_obj.get_terms()

    # initial state
    phase_space.set_q(init_qp[:,0,:,:])
    phase_space.set_p(init_qp[:,1,:,:])
    phase_space.set_boxsize(boxsize)
    #init_u = terms[0].energy(phase_space)
    init_e = hamiltonian_obj.total_energy(phase_space)

    # strike append state
    phase_space.set_q(qp_strike_append[:,0,:,:])
    phase_space.set_p(qp_strike_append[:,1,:,:])
    #strike_append_u = terms[0].energy(phase_space)
    strike_append_e = hamiltonian_obj.total_energy(phase_space)

    del_e = strike_append_e - init_e
    print('del e', del_e)

