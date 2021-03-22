import torch
from utils.paired_distance_reduce import paired_distance_reduce

class pb:

    _obj_count = 0

    def __init__(self):

        pb._obj_count += 1
        assert(pb._obj_count == 1), type(self).__name__ + ' has more than one object'

    def adjust_reduced(self,q): # use in Lennard-Jones class
         self.adjust_real(q,boxsize=1)

    def adjust_real(self, q, boxsize): #use in verlet and other classes

         indices = torch.where(torch.abs(q)>0.5*boxsize)
         shift = torch.round(q[indices] / boxsize) * boxsize
         # print('q i ',q[indices])
         # print('shift ',shift)
         q[indices] = q[indices] - torch.round(q[indices] / boxsize) * boxsize

    def debug_pbc(self, q, boxsize):

        bool = torch.abs(q) > 0.5 * boxsize
        # print('debug_pbc', bool.any())
        # print('q',q)

        if bool.any() == True: # if values have any above condition, return true.
            #print('q', q)
            index = torch.where(torch.abs(q) > 0.5 * boxsize)
            debug = q[index]
            # print('index', index)
            # print('debug_pbc',debug)
            print('debug pbc q',q)
            print('pbc not applied')

            # raise ValueError('pbc not applied')

    def debug_pbc_bool(self, q, boxsize):

        bool_ = torch.abs(q) > 0.5 * boxsize
        # print('debug_pbc', bool.any())
        # print('q',q)

        return bool_

    def debug_nan_bool(self, q, p):

        if (torch.isnan(q).any()) or (torch.isnan(p).any()):

            bool_ = torch.where(torch.isnan(q))
            print('debug nan q',q)

            return bool_

    def debug_pbc_reduced(self,q):

        index = torch.where(torch.abs(q)>0.5)
        debug = q[index]
        if debug.any():
            print('debug_pbc_reduced',q)
            raise ValueError('pbc reduced not applied')

    def debug_pbc_max_distance(self,q):

        boxsize = 1
        max_distance = torch.sqrt(boxsize/2. * boxsize/2. + boxsize/2. * boxsize/2.)
        index = torch.where(torch.abs(q)>max_distance)
        debug = q[index]
        if debug.any():
            print('debug_pbc_max_distance',q)
            raise ValueError('pbc reduced max distnace not applied')

    def paired_distance_reduced(self,q, nparticle, DIM):

        # print("==pb==")
        # print('q',q)
        # print('q', q.shape)
        qlen = q.shape[1]
        q0 = torch.unsqueeze(q,dim=1)
        qm = torch.repeat_interleave(q0,qlen,dim=1)

        qt = qm.permute(0,2,1,3)
        # print('qt', qt)
        # print('qm', qm)
        dq = qt - qm
        # print('dq', dq)

        indices = torch.where(torch.abs(dq)>0.5)
        dq[indices] = dq[indices] - torch.round(dq[indices])
        # print('dq pbc',dq)

        dq_reduced_index = paired_distance_reduce.get_indices(dq.shape)
        dq_flatten = paired_distance_reduce.reduce(dq, dq_reduced_index)
        # print('dq_flatten', dq_flatten)
        dq_flatten = dq_flatten.reshape((q.shape[0], nparticle, nparticle - 1, DIM))
        # print('dq_flatten', dq_flatten)
        # print('dq_flatten', dq_flatten.shape)
        dd = torch.sqrt(torch.sum(dq_flatten * dq_flatten, dim=-1))
        # print(dd)
        # print(dd.shape)

        return dq_flatten, dd

