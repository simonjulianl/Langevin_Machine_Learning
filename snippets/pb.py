import numpy as np

#==============================================
# range of q is always ([-0.5,0.5]x[-0.5,0.5]) - R^2
class periodic_bc:

    # q is a list of all particle position
    def adjust(self,q):

         indices = np.where(q>0.5)
         q[indices] = q[indices] - 1.0
         indices = np.where(q<-0.5)
         q[indices] = q[indices] + 1.0

    # returns pair distances between two particles
    # return a symmetric matrx
    def paired_distance(self,q):

        qlen = q.shape[0]
        q0 = np.expand_dims(q,axis=0)
        qm = np.repeat(q0,qlen,axis=0)
        qt = np.transpose(qm,axes=[1,0,2]) 
        dq = qm - qt
        print('dq - raw ')
        print(dq)

        indices = np.where(np.abs(dq)>0.5)
        dq[indices] = dq[indices] - np.copysign(1.0, dq[indices])
        print('dq - adjust ')
        print(dq)

        dd = np.sqrt(np.sum(dq*dq,axis=2))
        print('dd sum ')
        print(dd)
        return dd
#==============================================
class free_end_bc:

    # q is a list of all particle position
    def adjust(self,q):
        return # since this is free end, do nothing

#==============================================
class verlet: # for example and simulation

    def move_particle(self,q,bc):

        print('before move ',q)
        q = q + [1,1] # for example, move all particles
        print('after move ',q)
        bc.adjust(q)
        print('after adjust ',q)

#==============================================

class pairwise_harmonic:

    def potential_energy(self,q,bc):

        paired_matrix = bc.paired_distance(q)
        k = 1.0/2.0
        energy = np.sum(k*paired_matrix*paired_matrix)*0.5
        return energy
#==============================================

if __name__=='__main__':


    np.random.seed(132)
    #q = (2*np.random.random([3,2])-1)
    q = np.asarray([ [-0.5,-0.5],[0.5,0.4] ])
    pb = periodic_bc()
    pb.adjust(q)
    #v = verlet()
    #v.move_particle(q,pb)
    print('initial q ',q)
    pb.paired_distance(q)
    h = pairwise_harmonic()
    e = h.potential_energy(q,pb)
    print('e ',e)
    
