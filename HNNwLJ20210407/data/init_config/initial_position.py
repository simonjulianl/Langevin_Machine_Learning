import glob
import torch
import math
import numpy as np

import matplotlib.pyplot as plt

if __name__ == '__main__':

    ''' plot of particles initial position in boundary each sample taken a few '''

    # seed the generator
    np.random.seed(73893)

    nparticle   = 2
    temp        = 0.04
    DIM         = 2
    rho         = 0.1
    boxsize     = math.sqrt(nparticle/rho)

    mode      = 'train'  # set train or valid or test to read file

    root_path   = 'n{}/'.format(nparticle)
    filename = glob.glob( root_path + 'run*/' + 'nparticle{}_new_nsim_rho0.1_T{}'.format(nparticle, temp) + '_pos_{}_sampled.pt'.format(mode))
    print('file dir :', filename)

    data_q, data_p =  torch.load(filename[0])  # filename is list so that take first index.

    # take a few nsamples randomly
    for index in np.random.choice(data_q.shape[0],2):

        sample_q = data_q[index]

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        plt.xlim(-0.5*boxsize,0.5*boxsize)
        plt.ylim(-0.5*boxsize,0.5*boxsize)

        plt.title(r'$\rho$={}, BoxSize={:.2f}, T={}'.format(rho, boxsize, temp), fontsize=15)

        print(index, 'sample')
        for i in range(nparticle):
            plt.plot(sample_q[i,0],sample_q[i,1],'o',markersize=30)
            print('particle', i, 'qx:', sample_q[i,0], 'qy:', sample_q[i,1])

        plt.show()
