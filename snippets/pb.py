import numpy as np
import matplotlib.pyplot as plt

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
        print(paired_matrix*paired_matrix)
        print(np.sum(1/2*paired_matrix*paired_matrix))
        energy = np.sum(k*paired_matrix*paired_matrix)*0.5
        return energy

class pairwise_LJ:

    def potential_energy(self,q,bc,BoxSize):

        paired_matrix = bc.paired_distance(q)
        print('paired_matrix',paired_matrix)
        #paired_matrix = eval('4 *  ((1/ (paired_matrix)) ** 12.0 - (1/(paired_matrix )) ** 6.0)')
        paired_matrix = eval('4 *  ((1/ (paired_matrix * BoxSize)) ** 12.0 - (1/(paired_matrix * BoxSize)) ** 6.0)')
        print('paired_matrix', paired_matrix)

        energy =np.nansum(paired_matrix)*0.5
        print(energy)

        return energy

class plot_pcbox:

    def pbbox(self, N):

        plt.cla()
        plt.xlim(-.5,.5)
        plt.ylim(-.5,.5)
        #plt.title(r'T={}'.format(T),fontsize=15)
        for i in range(N):
            #print('for',q[i])
            plt.plot(q[i,0],q[i,1],'o',markersize=15)
        #plt.savefig('./filesave/' + r'N{}_T{:.2f}_pos.png'.format(N,T))
        #plt.close()
        plt.show()
#==============================================

if __name__=='__main__':


    #np.random.seed(29999)
    q = (2*np.random.random([4,2])-1)
    #q = np.asarray([[ 0.4887738, -0.84314605],[-0.10962522,-0.94944091]])  # 2 particles in 2 dimension
    N, DIM = q.shape
    print('initial q ', q)
    pb = periodic_bc()
    pb.adjust(q)
    print('initial_move q ',q)
    plot_pcbox().pbbox(N)

    pb.paired_distance(q)
    print('print')
    h = pairwise_LJ()
    e = h.potential_energy(q,pb,BoxSize=3.16)
    print('e ',e)
    
