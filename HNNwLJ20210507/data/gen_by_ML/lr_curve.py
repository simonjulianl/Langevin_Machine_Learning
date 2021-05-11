import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import sys
import math

# epoch:0 train_loss:1 valid_loss:2 train_dq/boxsz:3 valid_dq/boxsz:4
# train_dq:5 valid_dq:6 train_dp:7 valid_dp:8 lr:9 time:10

if __name__ == '__main__':

    argv = sys.argv

    if len(argv) != 7:
        print('usage <programe> <basename> <lr> <nsamples e.g. "600,000/120,000"> <batch_size 12,000> <lr decay> <titan>')
        quit()

    basename            = argv[1]
    lr_scheduler        = int(argv[2])
    nsamples            = argv[3]
    batch_size          = int(argv[4])
    lr_decay            = argv[5]
    titan               = argv[6]

    # paramters
    nsamples = nsamples
    batch_size = batch_size
    nparticle = 4
    long_tau = 0.1
    short_tau = "1e-4"
    rho = 0.1
    lr = 0.01
    lr_scheduler = lr_scheduler
    epochs = lr_decay
    NN = "5->128->128->16->2"
    optim = 'SGD'
    activation = 'tanh'
    time = '250'
    titan = titan

    boxsize = math.sqrt(nparticle/rho)
    data = np.genfromtxt( basename +'/' + basename + '_loss.txt')

    x = data[:, 0]
    y = data[:, lr_scheduler]

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    # ax2.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
    ax2.plot(x, y, 'blue', label='train', zorder=2)
    ax2.set_xlabel('epoch', fontsize=30)
    ax2.set_ylabel('lr', fontsize=30)
    ax2.tick_params(labelsize=20)
    # ax2.set_ylim([1.0986,1.0988])
    ax2.legend(loc='upper right', fontsize=20)
    plt.title(lr_decay, fontsize=20)
    anchored_text = AnchoredText(
        'nsamples={} batch size={} nparticle={} boxsize={:.3f} \nlarge time step = {}, short time step {} \nNN input 5  output 2 (F_x, F_y) opt {} lr {} {} \nactivation tanh() NN {} time per epoch= {}s titan {}'.format(
            nsamples, batch_size, nparticle, boxsize, long_tau, short_tau, optim, lr, lr_decay, NN, time, titan),
        loc='upper left', prop=dict(fontweight="normal", size=12))
    ax2.add_artist(anchored_text)
    plt.grid()
    plt.show()
    plt.close()
