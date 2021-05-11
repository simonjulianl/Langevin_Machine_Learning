from grids import grids
import plot_grids
from phase_space import phase_space
from phi_fields import phi_fields
from data_io import data_io
from noML_hamiltonian import noML_hamiltonian

import torch

if __name__ == '__main__':

    seed = 9372211
    torch.manual_seed(seed)

    nsamples = 2
    ng = 16 # gridL
    # L =  4. # boxsize

    data_io = data_io()
    noML_hamiltonian = noML_hamiltonian()
    phase_space = phase_space()

    init_qp, _, _, boxsize = data_io.read_trajectory_qp('n2T0.1seed123nsamples2.pt')

    # init_qp, _, _, _ = data_io.read_trajectory_qp('n4T0.03seed6325nsamples10.pt')
    # init_qp.shape = [nsamples, (q, p), 1, nparticle, DIM]

    # init_q = torch.squeeze(init_qp[33:33+nsamples,0,:,:,:], dim=1)
    # init_q = torch.tensor([[[-1.9, -1.3], [1.9, -0.9]], [[-1.9, -1.3], [1.2, -0.9]]])
    # init_q.shape = [nsamples, nparticle, DIM]

    init_q = torch.squeeze(init_qp[50:50+nsamples, 0, :, :, :], dim=1)
    init_p = torch.squeeze(init_qp[50:50 + nsamples, 0, :, :, :], dim=1)
    phase_space.set_q(init_q)
    phase_space.set_p(init_p)
    phase_space.set_boxsize(boxsize)

    phi_fields = phi_fields(ng, noML_hamiltonian)
    phi_fields.gen_phi_fields(phase_space)
    quit()
    print('q', init_q)

    i0j0, i0j1, i1j0, i1j1, x0y0, x0y1, x1y0, x1y1 = grids.get_nearest_coord(init_q)
    # i0j0.shape and x0y0.shape is [ batch_size, nparticle, DIM ]
    print(i0j0, i0j1, i1j0, i1j1, x0y0, x0y1, x1y0, x1y1)

    grids_list = plot_grids.build_grid_list(init_q, L, ng)
    plot_grids.show_grids_nparticles(grids_list,init_q)
    grids_list = grids_list.reshape((ng,ng,2))
    # print(grids_list[0,2])