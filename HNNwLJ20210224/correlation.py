import torch
import math
import matplotlib.pyplot as plt

nparticle = 4
DIM = 2
T = 0.04
samples = 1000
time_step = 0.1
rho = 0.1
boxsize = math.sqrt(nparticle/rho)
max_ts = 20
tau_long = 0.01
tau_short = 0.001

gold_standard = torch.load('./gold_standard/nparticle{}_T{}_ts0.001_iter20000_vv_1000sampled.pt'.format(nparticle,T))
tau_large = torch.load('./gold_standard/nparticle{}_T{}_ts0.01_iter2000_vv_1000sampled.pt'.format(nparticle,T)) # HNN
#tau_large = np.load('Langevin_Machine_Learning/init/N{}_T{}_ts{}_iter{}_vv_gm0.0_1000sampled_predicted_1.npy'.format(N_particle,T,time_step,iterations)) # MD

print('boxsize',boxsize)
L_h = boxsize/2.
q_max = math.sqrt(L_h*L_h + L_h*L_h)
print('Maximum distance r = {}, r^2 = {}'.format(q_max,q_max*q_max))

gs_q = gold_standard[0]
q    = tau_large[0]

gs_p = gold_standard[1]
p    = tau_large[1]

sample = gs_q.shape[1]
print('sample',sample)

del_q = torch.pow((gs_q - q),2)
del_p =  torch.pow((gs_p - p),2)

# print('----gs_q----')
# print(gs_q[:,0])
# print('----q----')
# print(q[:,0])
# print('--gs_q - q--')
# print(gs_q[:,0] - q[:,0])
# print('----del_q----')
# print(del_q[:,0])

# print('----gs_p----')
# print(gs_p)
# print('----p----')
# print(p)
# print('--gs_p - p--')
# print(gs_p - p)
# print('----del_p----')
# print(del_p)

print(del_q.shape,del_p.shape)

avg_del_q_particle = torch.sum(del_q, dim=2) / nparticle
# print('----avg_del_q_sample each component----')
# print(avg_del_q_sample)
# print(avg_del_q_sample.shape)

avg_del_q_particle = torch.sum(avg_del_q_particle, dim=2)
print('del_q per particle', avg_del_q_particle.shape)

avg_del_p_particle = torch.sum(del_p, dim=2) / nparticle
# print('----avg_del_p_sample each component----')
# print(avg_del_p_sample)
# print(avg_del_p_sample.shape)

avg_del_p_particle = torch.sum(avg_del_p_particle , dim=2)
print('del_p per particle', avg_del_p_particle.shape)

del_qp_particle = avg_del_q_particle + avg_del_p_particle
avg2_del_qp_particle_sample = torch.sum( torch.pow(del_qp_particle , 2), dim=1) / sample
print(avg2_del_qp_particle_sample.shape)

avg_del_q_particle_sample = torch.sum(avg_del_q_particle, dim=1) / sample
# print('----avg_del_q_sample_particle sum components----')
# print(avg_del_q_sample_particle)

avg_del_p_particle_sample = torch.sum(avg_del_p_particle, dim=1) / sample
# print('----avg_del_p_sample_particle sum components----')
# print(avg_del_p_sample_particle)

avg_del_qp_particle_sample = avg_del_q_particle_sample + avg_del_p_particle_sample
print('----avg_del_qp_sample_particle -----')
# print(avg_del_qp_particle_sample)
print(avg_del_qp_particle_sample.shape)

std_del_qp_particle_sample = (avg2_del_qp_particle_sample - avg_del_qp_particle_sample*avg_del_qp_particle_sample)**0.5
print('----std_del_qp_particle_sample -----')
print(std_del_qp_particle_sample.shape)

fig = plt.figure()
t = torch.arange(0., max_ts + time_step, time_step)
plt.suptitle(r'boxsize {:.4f}, Maximum distance $r = {:.4f}, r^2 = {:.4f}$'.format(boxsize,q_max,q_max*q_max)
             + '\nTemp = {}, pair with time step = {}, Maximum time step = {}'.format(T, time_step, max_ts)
             + '\n' + r'gold standard $\tau^\prime$ = {}, time step compared with gold standard $\tau$ = {}'.format(tau_short, tau_long))

fig.add_subplot(2,2,1)
plt.plot(t,avg_del_q_particle_sample, label = 'Distance metric')
plt.ylabel(r'$\Delta q^{\tau,{\tau}^\prime}$',fontsize=15)

fig.add_subplot(2,2,2)
plt.ylabel(r'$\Delta p^{\tau,{\tau}^\prime}$',fontsize=15)
plt.plot(t,avg_del_p_particle_sample, label = 'Distance metric')

fig.add_subplot(2,1,2)
#plt.title('Temp {}; samples {}; iterations {}; Training epochs {};'.format(T,samples,iterations,epochs)+r'$\tau={} ; \tau^\prime=0.001$'.format(time_step))
plt.xlabel('time',fontsize=16)
plt.ylabel(r'$\Delta^{\tau,{\tau}^\prime}$',fontsize=18)
#plt.ylim(-0.005,0.1)
#plt.ylim(-0.01,0.89)
#plt.ylim(-0.01,0.1)
#plt.xlim(0,2.5)
plt.plot(t,avg_del_qp_particle_sample, label = 'Distance metric')
plt.tick_params(axis='y',labelsize=16)
plt.show()

