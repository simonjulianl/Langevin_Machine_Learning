from hex_grids import hex_grids
from data_io import data_io
from phase_space import phase_space
from noML_hamiltonian import noML_hamiltonian
from phi_fields import phi_fields
import torch

if __name__ == '__main__':

    seed = 9372211
    torch.manual_seed(seed)

    nsamples = 1

    data_io = data_io()
    noML_hamiltonian = noML_hamiltonian()
    phase_space = phase_space()

    init_qp, _, _, boxsize = data_io.read_trajectory_qp('n2T0.1seed123nsamples2.pt')
    # init_qp, _, _, boxsize = data_io.read_trajectory_qp('n4T0.03seed6325nsamples10.pt')
    # init_qp.shape = [nsamples, (q, p), 1, nparticle, DIM]

    # init_q = torch.squeeze(init_qp[33:33+nsamples,0,:,:,:], dim=1)
    # init_q = torch.tensor([[[-1.9, -1.3], [1.9, -0.9]], [[-1.9, -1.3], [1.2, -0.9]]])
    # init_q.shape = [nsamples, nparticle, DIM]

    init_q = torch.squeeze(init_qp[50:50+nsamples, 0, :, :, :], dim=1)
    init_p = torch.squeeze(init_qp[50:50 + nsamples, 0, :, :, :], dim=1)

    phase_space.set_q(init_q)
    phase_space.set_p(init_p)
    phase_space.set_boxsize(boxsize)

    print('boxsize', boxsize)
    print('q position', init_q)

    phi_fields = phi_fields(noML_hamiltonian)
    phi = phi_fields.gen_phi_fields(phase_space)
    #phi.shape is [nsamples, nparticle, grids18]
