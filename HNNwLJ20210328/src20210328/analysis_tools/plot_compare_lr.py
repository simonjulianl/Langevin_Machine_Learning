import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import math

nsamples = 20900
nparticle = 4
long_tau = 0.1
short_tau = 0.001
rho = 0.1
lr1 = 0.003
lr2 = 0.0001
hidden = 32
optim = 'Adam'
activation = 'tanh'
start = 0
end = 300

boxsize = math.sqrt(nparticle/rho)
data1 = np.genfromtxt('nsamples{}_nparticle{}_tau{}_{}_{}_lr{}_h{}_{}_loss.txt'.format(nsamples,nparticle,long_tau,short_tau,optim, lr1,hidden,activation))
data2 = np.genfromtxt('nsamples{}_nparticle{}_tau{}_{}_{}_lr{}_h{}_{}_loss.txt'.format(nsamples,nparticle,long_tau,short_tau,optim, lr2,hidden,activation))
#data = np.genfromtxt('nsamples{}_nparticle{}_tau{}_loss.txt'.format(nsamples,nparticle,long_tau,short_tau,lr,hidden))

x = range(start,end)
loss1 = data1[:,1]
val_loss1 = data1[:,2]
loss2 = data2[:,1]
val_loss2 = data2[:,2]

loss1 = loss1[start:end]
val_loss1 = val_loss1[start:end]
loss2 = loss2[start:end]
val_loss2 = val_loss2[start:end]

fig2 =plt.figure()
ax2 = fig2.add_subplot(111)
#ax2.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
ax2.plot(x,loss1,'blue',label='lr={} train'.format(lr1),zorder=2)
ax2.plot(x,loss2,'green',label='lr={} train'.format(lr2),zorder=2)
ax2.plot(x,val_loss1,'orange',label='lr={} valid'.format(lr1),zorder=2)
ax2.plot(x,val_loss2,'red',label='lr={} valid'.format(lr2),zorder=2)
ax2.set_xlabel('epoch',fontsize=30)
ax2.set_ylabel('Loss',fontsize=30)
ax2.tick_params(labelsize=20)
#ax2.set_ylim([1.0986,1.0988])
ax2.legend(loc='upper right',fontsize=20)
#plt.title('nsamples={} nparticle={} boxsize={:.3f}'.format(nsamples,nparticle, boxsize),fontsize=20)
anchored_text = AnchoredText('nsamples={} nparticle={} boxsize={:.3f} \nlarge time step = 0.1, short time step 0.001 \nnn input 5  output 2 each batch {} data split ratio = train 19000 / valid 1900 \nopt {} lr {},{} nn 2 hidden layers each {} units\nactivation tanh() '.format(nsamples,nparticle, boxsize,nparticle*(nparticle-1),optim,lr1,lr2,hidden), loc= 'upper left', prop=dict(fontweight="normal", size=16))
ax2.add_artist(anchored_text)
plt.grid()
plt.show()
plt.close()
