from phase_space.phase_space         import phase_space
from integrator.linear_integrator    import linear_integrator
from integrator.methods              import linear_velocity_verlet
from hamiltonian.noML_hamiltonian    import noML_hamiltonian
from utils.data_io                   import data_io
from utils.show_graph                import show_graph

import sys
import torch

def load_anydata(infile1):
    ''' load train or valid or train data'''
    init_qp, _, _, boxsize = data_io.read_trajectory_qp(infile1)
    # init_qp_train.shape = [nsamples, (q, p), 1, nparticle, DIM]

    init_q = torch.squeeze(init_qp[:,0,:,:,:], dim=1)
    init_p = torch.squeeze(init_qp[:,1,:,:,:], dim=1)

    phase_space.set_q(init_q)
    phase_space.set_p(init_p)
    phase_space.set_boxsize(boxsize)

    return init_q.shape, boxsize

if __name__ == '__main__':

    ''' potential energy fluctuations for mc steps that appended to nsamples and 
        histogram of nsamples generated from mc simulation'''

    # run something like this
    # python check_gen_by_mc.py all ../data/gen_by_MC/train/filename ../data/gen_by_MC/valid/filename

    argv = sys.argv

    temp      = argv[1]
    infile1   = argv[2]
    infile2   = argv[3]

    # parameters
    temp         = temp

    phase_space = phase_space()
    hamiltonian_obj = noML_hamiltonian()
    linear_integrator_obj = linear_integrator(linear_velocity_verlet.linear_velocity_verlet, linear_velocity_verlet.linear_velocity_verlet_backward)

    terms = hamiltonian_obj.get_terms()

    # train data
    q_train_shape, boxsize = load_anydata(infile1)  # q_train_shape is [nsamples, nparticle, DIM]
    u_term1 = terms[0].energy(phase_space)

    # valid data
    q_valid_shape, _ = load_anydata(infile2)  # q_train_shape is [nsamples, nparticle, DIM]
    u_term2 = terms[0].energy(phase_space)

    # show_graph.u_fluctuation(u_term1, temp, q_train_shape[0], 'train')
    # show_graph.u_fluctuation(u_term2, temp, q_valid_shape[0], 'valid')
    show_graph.u_distribution4nsamples(u_term1, u_term2, temp, q_valid_shape[1], boxsize, q_train_shape[0], q_valid_shape[0])


