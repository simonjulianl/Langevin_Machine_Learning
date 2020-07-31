import numpy as np

#==============================================
# range of q is always ([-0.5,0.5]x[-0.5,0.5]) - R^2
class periodic_bc:

    def adjust(self,q):

         indices = np.where(q>0.5)
         q[indices] = q[indices] - 1.0
         indices = np.where(q<-0.5)
         q[indices] = q[indices] + 1.0

    def adjust_real(self,q,boxsize):

         indices = np.where(q>0.5*boxsize)
         q[indices] = q[indices] - 1.0 *boxsize
         indices = np.where(q<-0.5*boxsize)
         q[indices] = q[indices] + 1.0*boxsize

    # returns pair distances between two particles
    # return a symmetric matrx
    def paired_distance(self,q):

        qlen = q.shape[0]
        q0 = np.expand_dims(q,axis=0)
        ##print('q0',q0)
        #print('q0', q0.shape)
        qm = np.repeat(q0,qlen,axis=0)
        #print('qm',qm)
        #print('qm', qm.shape)
        qt = np.transpose(qm,axes=[1,0,2])
        #print('qt',qt)
        dq = qm - qt
        ##print('dq - raw ')
        ##print(dq)

        indices = np.where(np.abs(dq)>0.5)
        dq[indices] = dq[indices] - np.copysign(1.0, dq[indices])
        ##print('dq - adjust ')
        ##print(dq)
        delta_q = np.sum(dq,axis=1)
        ##print('delta_q',delta_q)
        #print('dq*dq')
        #print(dq*dq)
        #print(np.sum(dq*dq,axis=2))
        dd = np.sqrt(np.sum(dq*dq,axis=2))
        ##print('dd sum ')
        ##print(dd)
        return delta_q, dd

