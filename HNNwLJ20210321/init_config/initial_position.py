import sys
import os
sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("../parameters"))

import torch
import matplotlib.pyplot as plt
from HNN.pair_wise_HNN import pair_wise_HNN
from HNN.models.pair_wise_MLP import pair_wise_MLP
from phase_space.phase_space import phase_space
import math
import numpy as np

if __name__ == '__main__':

    np.random.seed(73893)

    temp = 0.04
    nsamples = 100
    DIM = 2
    nparticle = 4
    rho = 0.1
    boxsize = math.sqrt(nparticle/rho)
    print('boxsize', boxsize)

    phase_space = phase_space()
    pair_wise_HNN_obj = pair_wise_HNN(pair_wise_MLP())

    noMLhamiltonian = super(type(pair_wise_HNN_obj), pair_wise_HNN_obj)
    terms = noMLhamiltonian.get_terms()

    data_q, data_p =  torch.load('nparticle{}_new_nsim_rho0.1_T{}_pos_train_sampled.pt'.format(nparticle,temp))
    print(data_q.shape)

    # plt.title('T={}'.format(temp),fontsize=15)
    # plt.plot(u_term,'k-', label = 'train : each {} mcs from 50 random samplings'.format(nsamples))
    # plt.xlabel('mcs',fontsize=20)
    # plt.ylabel(r'$U_{ij}$',fontsize=20)
    # plt.legend()
    # plt.show()

    for index in np.random.choice(data_q.shape[0],20):

        # print(index)
        sample_q = data_q[index]
        sample_p = data_p[index]

        sample_q = torch.unsqueeze(sample_q, dim=0)

        phase_space.set_q(sample_q)
        u_term = terms[0].energy(phase_space)
        # phase_space.set_p(sample_p)

        _, dd = phase_space.paired_distance_reduced(sample_q[0], nparticle, DIM)
        boxsize_ = torch.tensor([boxsize] * nparticle * (nparticle - 1))
        boxsize_ = boxsize_.reshape(nparticle, (nparticle - 1))
        # print(dd)
        # print(dd * boxsize_)

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        plt.xlim(-0.5*boxsize,0.5*boxsize)
        plt.ylim(-0.5*boxsize,0.5*boxsize)
        plt.title(r'$\rho$={}, BoxSize={:.2f}, T={}, U={:.2f}'.format(rho,boxsize,temp,u_term[0]),fontsize=15)
        for i in range(nparticle):
            # print(sample_q[0][i,0],sample_q[0][i,1])
            plt.plot(sample_q[0][i,0],sample_q[0][i,1],'o',markersize=30)
        #plt.savefig('./filesave/' + r'N{}_T{:.2f}_pos.png'.format(N_particle,T))

        plt.show()
