import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import argparse

parser = argparse.ArgumentParser(description='hello command line')
parser.add_argument('--method',   required=True, type=str, help='METHOD')
parser.add_argument('--sample',   required=True, type=int, help='SAMPLE')
parser.add_argument('--Temp',   required=True, type=float, help='Temperature')
parser.add_argument('--ts',   required=True, type=float, help='time step')
pars = vars(parser.parse_args())

method = pars['method']
sample = pars['sample']
T = pars['Temp']
ts = pars['ts']

DIM = 2
N_particle = 4
rho = 0.2
BoxSize = np.sqrt(N_particle/rho)

data_q, data_p =  np.load('N{}_T{}_ts{}_{}_sampled.npy'.format(N_particle,T,ts,method))
print(data_q.shape)
print(data_p.shape)

sample_q = data_q[:,sample]
sample_p = data_p[:,sample]
#print(sample_q[:10])

def LJ(q):
    #print(q)
    _epsilon = 1.
    _sigma = 1.
    expression = '4 * {0} * (({1}/ q) ** 12.0 - ({1}/q) ** 6.0)'.format(_epsilon, _sigma)
    lj = eval(expression)
    lj[~np.isfinite(lj)] = 0
    #print('lj',lj)
    term = np.sum(lj) * 0.5
    #print(term)
    return term

def kinetic_energy(p):
    #print(p)
    ene_kin_ = 0.0
    m = 1
    for j in range(N_particle):
        #print(p[j,:])
        ene_kin_ += 0.5 * m * np.sum(np.multiply(p[j, :], p[j, :]), axis=0)

    return ene_kin_

def instantaneous_temp(ene_kin_):
    #print('KE',ene_kin_)
    m = 1
    for j in range(N_particle):
        #print(p[j,:])
        ene_kin_aver = 1.0 * ene_kin_ / N_particle
        temperature = 2.0 / DIM * ene_kin_aver

    return temperature

def paired_distance(q,BoxSize):
    qlen = q.shape[0]
    q0 = np.expand_dims(q,axis=0)
    qm = np.repeat(q0,qlen,axis=0)
    qt = np.transpose(qm,axes=[1,0,2])
    dq = qm -qt

    indices = np.where(np.abs(dq)>0.5*BoxSize)
    dq[indices] = dq[indices] - np.copysign(1.0*BoxSize, dq[indices])
    dd = np.sqrt(np.sum(dq*dq,axis=2))
    #print('dd',dd)

    return dd

def init():
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    line4.set_data([], [])
    return [line1,line2,line3,line4]

def animate(i):
    print(i)
    q1 = sample_q[i,0]
    q2 = sample_q[i,1]
    q3 = sample_q[i,2]
    q4 = sample_q[i,3]
    #print(q1,q2,q3,q4)
    dd = paired_distance(sample_q[i],BoxSize)
    r21, r31, r41, r32, r42, r43 = dd[0][1],dd[0][2],dd[0][3],dd[1][2],dd[1][3],dd[2][3]
    #print(r21, r31, r41, r32, r42, r43)
    PE = LJ(dd)

    p = sample_p[i]
    KE = kinetic_energy(p)
    T_t = instantaneous_temp(KE)
    line1.set_data(q1[0],q1[1])
    line2.set_data(q2[0],q2[1])
    line3.set_data(q3[0],q3[1])
    line4.set_data(q4[0],q4[1])
    #ax.set_title(r'$\rho$={}; BoxSize={:.2f}; T={}; ts={}; t={:.3f}; rij={:.3f}; PE={:.3f}; KE={:.3f}'.format(rho,BoxSize,T,ts,i*ts,dd,PE,KE),fontsize=15)
    ax.set_title(r'$\rho$={}; BoxSize={:.2f}; T={}; ts={}; t={:.3f};'.format(rho, BoxSize, T,ts, i * ts) + '\n' + 'r21={:.3f};r31={:.3f};r41={:.3f};r32={:.3f};r42={:.3f};r43={:.3f};'.format(r21,r31, r41, r32, r42, r43) + '\n' + 'PE={:.3f}; KE={:.3f}; T(t)={:.3f}'.format(PE,KE,T_t), fontsize=10)

    return [line1,line2,line3,line4]

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
line1, = ax.plot([], [],'o',color = 'r', markersize=110)
line2, = ax.plot([], [],'o',color = 'g', markersize=110)
line3, = ax.plot([], [],'o',color = 'b', markersize=110)
line4, = ax.plot([], [],'o',color = 'y', markersize=110)
ax.set_xlim(-0.5*BoxSize,0.5*BoxSize)
ax.set_ylim(-0.5*BoxSize,0.5*BoxSize)

myani = FuncAnimation(fig, animate,init_func=init, frames=sample_q.shape[0], interval=60,repeat=False,blit=False)
output ="{}_s{}_T{}_ts{}".format(method,sample,T,ts)
#myani.save(output+'.mp4', fps=15)
myani.save(output+'.mp4')
#plt.show()
plt.close()


