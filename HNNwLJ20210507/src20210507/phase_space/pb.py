import torch
from utils.get_paired_distance_indices import get_paired_distance_indices

class pb:

    ''' pb class is parent class that has phase_space child class.
        adjust particles in the boundary and calculate pair-wise distance btw two particles
    '''

    _obj_count = 0

    def __init__(self):

        pb._obj_count += 1
        assert(pb._obj_count <= 2), type(self).__name__ + ' has more than two objects'
        # one phase space object for the whole code
        # the other phase space object only use as a copy in lennard-jones class in dimensionless
        # form
        print('pb initialized')

    def adjust_real(self, q, boxsize):

        ''' function to put particle back into boundary and use this function in integrator method

        Parameters
        ----------
        q : torch.tensor
                shape is [nsamples, nparticle, DIM]
        boxsize : float

        Returns
        ----------
        adjust q in boundary condition
        shape is [nsamples, nparticle, DIM]
        '''

        indices = torch.where(torch.abs(q)>0.5*boxsize)
        q[indices] = q[indices] - torch.round(q[indices] / boxsize) * boxsize


    def paired_distance_reduced(self, q, nparticle, DIM):

        ''' function to calculate reduced distance btw two particles

        Parameters
        ----------
        q : torch.tensor
                shape is [nsamples, nparticle, DIM]
        qlen : nparticle
        q0 : new tensor with a dimension of size one inserted at the specified position (dim=1)
        qm : repeated tensor which has the same shape as q0 along with dim=1
        qt : permutes the order of the axes of a tensor
        dq : pair-wise distance btw two particles
        dq_reshape : obtain dq of non-zero indices
        dd : sum over DIM

        Returns
        ----------
        dq_reshape : pair-wise distances each DIM per nparticle
        dd : sum over DIM each particle
        '''

        qlen = q.shape[1]
        q0 = torch.unsqueeze(q,dim=1)
        # q0.shape is [nsamples, 1, nparticle, DIM]
        qm = torch.repeat_interleave(q0,qlen,dim=1)
        # qm.shape is [nsamples, nparticle, nparticle, DIM]

        qt = qm.permute(get_paired_distance_indices.permute_order)
        # qt.shape is [nsamples, nparticle, nparticle, DIM]

        dq = qt - qm
        # dq.shape is [nsamples, nparticle, nparticle, DIM]

        indices = torch.where(torch.abs(dq)>0.5)
        dq[indices] = dq[indices] - torch.round(dq[indices])

        dq_reduced_index = get_paired_distance_indices.get_indices(dq.shape)
        dq_flatten = get_paired_distance_indices.reduce(dq, dq_reduced_index)
        # dq_flatten.shape is [nsamples x nparticle x (nparticle - 1) x DIM]

        dq_reshape = dq_flatten.reshape((q.shape[0], nparticle, nparticle - 1, DIM))
        # dq_reshape.shape is [nsamples, nparticle, (nparticle - 1), DIM]

        dd = torch.sqrt(torch.sum(dq_reshape * dq_reshape, dim=-1))
        # dd.shape is [nsamples, nparticle, (nparticle - 1 )]

        return dq_reshape, dd

