import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import math

nsamples = 400
nparticle = 4
tau = 0.1
rho = 0.1
lr = 1e-05
start = 5
end = 250
time = 4

boxsize = math.sqrt(nparticle/rho)
data = np.genfromtxt('nsamples{}_nparticle{}_tau{}_lr{}_loss.txt'.format(nsamples,nparticle,tau,lr))

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
plt.grid()
plt.show()
plt.close()
