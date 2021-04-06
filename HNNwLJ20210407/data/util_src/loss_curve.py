import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import math

if __name__ == '__main__':

    # paramters
    nsamples = 20900
    nparticle = 16
    long_tau = 0.1
    short_tau = 0.001
    rho = 0.1
    lr = 0.0001
    hidden = 32
    optim = 'Adam'
    activation = 'tanh'
    start = 0
    end =  42
    time = '9000'
    titan = '?'

    boxsize = math.sqrt(nparticle/rho)
    data = np.genfromtxt('nsamples{}_nparticle{}_tau{}_{}_{}_lr{}_h{}_{}_loss.txt'.format(nsamples,nparticle,long_tau,short_tau,optim, lr,hidden,activation))

    x = range(start,end)
    loss = data[:,1]
    val_loss = data[:,2]

    loss = loss[start:end]
    val_loss = val_loss[start:end]

    fig2 =plt.figure()
    ax2 = fig2.add_subplot(111)
    #ax2.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
    ax2.plot(x,loss,'blue',label='train',zorder=2)
    ax2.plot(x,val_loss,'orange',label='valid',zorder=2)
    ax2.set_xlabel('epoch',fontsize=30)
    ax2.set_ylabel('Loss',fontsize=30)
    ax2.tick_params(labelsize=20)
    #ax2.set_ylim([1.0986,1.0988])
    ax2.legend(loc='upper right',fontsize=20)
    #plt.title('nsamples={} nparticle={} boxsize={:.3f}'.format(nsamples,nparticle, boxsize),fontsize=20)
    anchored_text = AnchoredText('nsamples={} nparticle={} boxsize={:.3f} \nlarge time step = 0.1, short time step 0.001 \nnn input 5  output 2 each batch {} data split = train 19000 / valid 1900 \nopt {} lr {} nn 2 hidden layers each {} units\nactivation tanh() time per epoch= {}s titan {}'.format(nsamples,nparticle, boxsize,nparticle*(nparticle-1),optim,lr,hidden,time,titan), loc= 'upper left', prop=dict(fontweight="normal", size=12))
    ax2.add_artist(anchored_text)
    plt.grid()
    plt.show()
    plt.close()
