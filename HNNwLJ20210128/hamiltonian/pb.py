import torch

class pb:

    def adjust_reduced(self,q): # use in Lennard-Jones class
         self.adjust_real(q,boxsize=1)

    def adjust_real(self,q,boxsize): #use in verlet and other classes

         indices = torch.where(torch.abs(q)>0.5*boxsize)
         q[indices] = q[indices] - torch.round(q[indices] / boxsize) * boxsize

    def debug_pbc(self,q,boxsize):

        bool = torch.abs(q) > 0.5*boxsize
        #print(bool.any())
        if bool.any() == True: # if values have any above condition, return true.
            index = torch.where(torch.abs(q)>0.5*boxsize)
            debug = q[index]
            print('debug_pbc',debug)
            raise ValueError('pbc not applied')

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
        qlen = q.shape[0]
        q0 = torch.unsqueeze(q,dim=0)
        qm = torch.repeat_interleave(q0,qlen,dim=0)
        qt = qm.permute(1,0,2)
        # print(qt)
        # print(qm)
        dq = qt - qm
        # print(dq)
        indices = torch.where(torch.abs(dq)>0.5)
        dq[indices] = dq[indices] - torch.round(dq[indices])
        # print('pbc',dq)
        dq = dq[dq.nonzero(as_tuple=True)].reshape(nparticle, nparticle - 1, DIM)
        # print(dq)
        dd = torch.sqrt(torch.sum(dq*dq,dim=2))
        # print(dd)

        return dq, dd

