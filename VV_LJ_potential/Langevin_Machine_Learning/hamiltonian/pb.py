import numpy as np

#==============================================
# range of q is always ([-0.5,0.5]x[-0.5,0.5]) - R^2
class periodic_bc:

    def adjust_reduced(self,q): # use in Lennard-Jones class
         self.adjust_real(q,boxsize=1)

    def adjust_real(self,q,boxsize): #use in verlet and other classes

         indices = np.where(np.abs(q)>0.5*boxsize)
         #q[indices] = q[indices] - 1.0 *boxsize
         q[indices] = q[indices] - np.round(q[indices] / boxsize) * boxsize

         # indices = np.where(q<-0.5*boxsize)
         # #q[indices] = q[indices] + 1.0 * boxsize
         # q[indices] = q[indices] - np.round(q[indices] / boxsize) * boxsize


    def debug_pbc(self,q,boxsize):
        #print('debug q',q)
        index = np.where(np.abs(q)>0.5*boxsize)
        debug = q[index]
        if debug.any():
            print('q',q)
            print('index',index)
            print('boxsize', boxsize)
            print('boxsize/2',0.5*boxsize)
            print('debug_pbc',debug)
            raise ValueError('pbc not applied')

    def debug_pbc_reduced(self,q):

        index = np.where(np.abs(q)>0.5)
        debug = q[index]
        if debug.any():
            print('debug_pbc_reduced',q)
            raise ValueError('pbc reduced not applied')

    def debug_pbc_max_distance(self,d,boxsize):
        #print('before debug',d*boxsize)
        max_distance = np.sqrt(boxsize/2.*boxsize/2. + boxsize/2.*boxsize/2.)
        index = np.where( d*boxsize > max_distance)
        d[index] = max_distance/boxsize
        #print('debug_pbc',d*boxsize)
        # if debug.any():
        #     print('debug_pbc_max_distance',d*boxsize)
        #     raise ValueError('pbc reduced max distnace not applied')

    # returns pair distances between two particles
    # return a symmetric matrx
    def paired_distance_reduced(self,q,q_adj):

        qlen = q.shape[0]
        #print('q',q)
        q0 = np.expand_dims(q,axis=0)
        #print('q0',q0)
        #print('q0', q0.shape)
        qm = np.repeat(q0,qlen,axis=0)
        #print('qm',qm)
        #print('qm', qm.shape)
        qt = np.transpose(qm,axes=[1,0,2])
        #print('qt',qt)
        dq = qt - qm
        #print('dq - raw ')
        #print(dq)

        indices = np.where(np.abs(dq)>0.5)
        dq[indices] = dq[indices] - np.copysign(1.0, dq[indices])
        #print('dq - adjust ')
        #print(dq)
        ##delta_q = np.sum(dq,axis=1)
        ##print('delta_q',delta_q)
        #print('dq*dq')
        #print(dq*dq)
        #print(np.sqrt(np.sum(dq*dq,axis=2)))
        dd = np.sqrt(np.sum(dq*dq,axis=2))
        #print('dd sum ')
        #print('dd',dd)
        nonzero = np.nonzero(dd)
        #print('before q adj',6.324555320336759*dd)
        dd[nonzero] = dd[nonzero] + q_adj
        #print('nonsum',6.324555320336759*dd)
        return dq, dd

