from scipy.spatial import distance
import numpy as np
import torch

def rij(d):
    # d12 = 1 / pow(d,12)
    # d6 = 1 / pow(d,6)
    # phi = 1. / d12 - 1. / d6
    phi = torch.sum((d*d),dim=0)
    return phi

if __name__ == '__main__':

    seed = 9372211
    torch.manual_seed(seed)

    nsamples = 2
    nparticle = 4
    ngrids = 8
    DIM = 2

    init_q = torch.rand((nsamples,nparticle,DIM))
    grids = torch.tensor([[[-1.,-1.],[-1., 0.],[0. ,1.],[1., 1.],[-1.,-1.5],[-1., 0.5],[0. ,1.],[1.5, 1.]]])
    grids = grids.expand((nsamples,ngrids,DIM))
    # grids.shape is [nsamples, nparticle*ngrids, DIM]

    fields = torch.zeros((nsamples, ngrids))

    for z in range(nsamples):

        d = distance.cdist(init_q[z].numpy(), grids[z].numpy(), 'euclidean')
        # ValueError: XA must be a 2-dimensional array.
        print(grids[z])
        d_torch = torch.from_numpy(d)
        # shape is [nparticle, ngrids]
        print(d_torch)
        phi_fields = rij(d_torch)
        # shape is [ngrids]

        print(phi_fields)

        fields[z] = phi_fields