import numpy as np

#==============================================
# range of q is always ([-0.5,0.5]x[-0.5,0.5]) - R^2
class pb:

    def adjust_reduced(self,q): # use in Lennard-Jones class
         self.adjust_real(q,boxsize=1)

    def adjust_real(self,q,boxsize): #use in verlet and other classes

         indices = np.where(np.abs(q)>0.5*boxsize)
         q[indices] = q[indices] - np.round(q[indices] / boxsize) * boxsize

    def debug_pbc(self,q,boxsize):

        index = np.where(np.abs(q)>0.5*boxsize)
        debug = q[index]
        if debug.any():
            print('debug_pbc',debug)
            raise ValueError('pbc not applied')

    def debug_pbc_reduced(self,q):

        index = np.where(np.abs(q)>0.5)
        debug = q[index]
        if debug.any():
            print('debug_pbc_reduced',q)
            raise ValueError('pbc reduced not applied')

    def debug_pbc_max_distance(self,q):

        boxsize = 1
        max_distance = np.sqrt(boxsize/2.*boxsize/2. + boxsize/2.*boxsize/2.)
        index = np.where(np.abs(q)>max_distance)
        debug = q[index]
        if debug.any():
            print('debug_pbc_max_distance',q)
            raise ValueError('pbc reduced max distnace not applied')

    # HK def paired_distance_reduced(self,q,q_adj):
    def paired_distance_reduced(self,q):

        # print("==pb==")
        # print('dimensionless', q)
        qlen = q.shape[0]
        q0 = np.expand_dims(q,axis=0)
        qm = np.repeat(q0,qlen,axis=0)
        qt = np.transpose(qm,axes=[1,0,2])
        dq = qt - qm

        indices = np.where(np.abs(dq)>0.5)
        dq[indices] = dq[indices] - np.copysign(1.0, dq[indices])
        dd = np.sqrt(np.sum(dq*dq,axis=2))

        return dq, dd

