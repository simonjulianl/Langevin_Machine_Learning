import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import math

nsamples = 400
nparticle = 4
long_tau = 0.1
short_tau = 0.001
rho = 0.1
lr = 0.001
hidden = 128
start = 0
end = 700
time = 18
titan = 2

boxsize = math.sqrt(nparticle/rho)
#data = np.genfromtxt('nsamples{}_nparticle{}_tau{}_{}_lr{}_h{}_loss.txt'.format(nsamples,nparticle,long_tau,short_tau,lr,hidden))
data = np.genfromtxt('nsamples{}_nparticle{}_tau{}_loss.txt'.format(nsamples,nparticle,long_tau,short_tau,lr,hidden))

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
anchored_text = AnchoredText('nsamples={} nparticle={} boxsize={:.3f} \nlarge time step = 0.1, short time step 0.01 \nnn input 5  output 2 each batch {} data split ratio = train 0.8 / valid 0.2 \nopt SGD lr {} nn 3 hidden layers each {} units\nactivation relu() time per epoch= {}s titan {}'.format(nsamples,nparticle, boxsize,nparticle*(nparticle-1),lr,hidden,time,titan), loc= 'center', prop=dict(fontweight="normal", size=16))
ax2.add_artist(anchored_text)
plt.grid()
plt.show()
plt.close()
