from phase_space.phase_space         import phase_space
from integrator.linear_integrator    import linear_integrator
from integrator.methods              import linear_velocity_verlet
from hamiltonian.noML_hamiltonian    import noML_hamiltonian
from utils.data_io                   import data_io
from utils.show_graph                import show_graph

import sys
import torch



if __name__ == '__main__':

    ''' potential energy fluctuations for mc steps that appended to nsamples and 
        histogram of nsamples generated from mc simulation'''

    # run something like this
    # python nsamples_generated_mc.py 4 0.04 ../data/init_config/n4train/filename ../data/init_config/n4valid/filename

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

    init_qp_train, _, _, boxsize = data_io.read_trajectory_qp(infile1)
    # init_qp_train.shape = [nsamples, (q, p), 1, nparticle, DIM]

    init_q_train = torch.squeeze(init_qp_train[:,0,:,:,:], dim=1)
    init_p_train = torch.squeeze(init_qp_train[:,1,:,:,:], dim=1)

    nsamples4train, nparticle, DIM = init_p_train.shape

    phase_space.set_q(init_q_train)
    phase_space.set_p(init_p_train)
    phase_space.set_boxsize(boxsize)

    u_term1 = terms[0].energy(phase_space)

    # nsamples from mc step generated for valid data
    init_qp_valid, _, _, _ = data_io.read_trajectory_qp(infile2)

    init_q_valid = torch.squeeze(init_qp_valid[:,0,:,:,:], dim=1)
    init_p_valid = torch.squeeze(init_qp_valid[:,1,:,:,:], dim=1)

    nsamples4valid, _, _ = init_p_valid.shape

    phase_space.set_q(init_q_valid)
    phase_space.set_p(init_p_valid)

    u_term2 = terms[0].energy(phase_space)

    show_graph.u_distribution4nsamples(u_term1, u_term2, temp, nparticle, boxsize, nsamples4train, nsamples4valid)


