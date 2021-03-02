import numpy as np
import matplotlib.pyplot as plt

# tau 0.1 till 10 iterations 100
# tau 0.01 till 10 iterations 1000
# tau 0.001 till 10 iterations 10000

N_particle =2
DIM = 2
T = 0.32
samples = 5000
iterations = 100
time_step = 0.1
interval = 100
rho = 0.2
epochs = 4000
BoxSize = np.sqrt(N_particle/rho)

gold_standard = np.load('Langevin_Machine_Learning/init/N{}_T{}_ts0.001_iter10000_vv_gm0.0_5000sampled.npy'.format(N_particle,T))
tau_large = np.load('Langevin_Machine_Learning/init/N{}_T{}_ts0.1_iter100_vv_gm0.0_5000sampled.npy'.format(N_particle,T)) # HNN
#tau_large = np.load('Langevin_Machine_Learning/init/N{}_T{}_ts{}_iter{}_vv_gm0.0_1000sampled_predicted_1.npy'.format(N_particle,T,time_step,iterations)) # MD

print('BoxSize',BoxSize)
L_h = BoxSize/2.
q_max = np.sqrt(L_h*L_h + L_h*L_h)
print('Maximum distance r = {}, r^2 = {}'.format(q_max,q_max*q_max))
print(gold_standard.shape)
print(tau_large.shape)

#gs_q_ = gold_standard[0,:300,:4]
gs_q_ = gold_standard[0]
# print(gs_q_[:21].shape)
# print('==============')
# print('every steps gs_q')
# print('initial',gs_q_[0])
# print('10step',gs_q_[100])
# print('20step',gs_q_[200])
print('======q======')
print('every 100 steps gs_q')
gs_q = gs_q_[0::interval,:samples]  #[0,10,20,30 ...,]
print(gs_q.shape)
# print('initial',gs_q[0])
# print('1step',gs_q[1])
# print('2step',gs_q[2])
print('==============')
print('every steps q')
q    = tau_large[0]
print('reshape N_particle x DIM q')
# need to reshape for prediction
#q    = q.reshape(-1,samples,N_particle,DIM)
print(q.shape)
# q    = q[:3,:4]
# print('initial',q[0])
# print('1step',q[1])
# print('2step',q[2])
print('==============')
print('======p======')
#gs_p_ = gold_standard[1,:300,:4]
gs_p_ = gold_standard[1]
# print('every 100 steps gs_p')
gs_p = gs_p_[0::interval,:samples]  #[0,10,20,30 ...,]
# print('initial',gs_p[0])
# print('1step',gs_p[1])
# print('2step',gs_p[2])
print('==============')
p    = tau_large[1]
print('reshape N_particle x DIM p')
# need to reshape for prediction
#p    = p.reshape(-1,samples,N_particle,DIM)
print(p.shape)
# p    = p[:3,:4]
# print('initial',p[0])
# print('1step',p[1])
# print('2step',p[2])
print('==============')

sample = gs_q.shape[1]
print('sample',sample)
print(gs_q.shape,q.shape)

# x
del_q = np.power((gs_q - q),2)
del_p =  np.power((gs_p - p),2)
# print('----gs_q----')
# print(gs_q)
# print('----q----')
# print(q)
# print('--gs_q - q--')
# print(gs_q - q)
# print('----del_q----')
# print(del_q)

# print('----gs_p----')
# print(gs_p)
# print('----p----')
# print(p)
# print('--gs_p - p--')
# print(gs_p - p)
# print('----del_p----')
# print(del_p)

print(del_q.shape,del_p.shape)

avg_del_q_particle = np.sum(del_q,axis=2) / N_particle
# print('----avg_del_q_sample each component----')
# print(avg_del_q_sample)
# print(avg_del_q_sample.shape)

avg_del_q_particle = np.sum(avg_del_q_particle,axis=2)
# print(avg_del_q_sample)

avg_del_p_particle = np.sum(del_p,axis=2) / N_particle
# print('----avg_del_p_sample each component----')
# print(avg_del_p_sample)
# print(avg_del_p_sample.shape)

avg_del_p_particle = np.sum(avg_del_p_particle ,axis=2)
print(avg_del_p_particle.shape)

del_qp_particle = avg_del_q_particle + avg_del_p_particle
avg2_del_qp_particle_sample = np.sum(np.power(del_qp_particle ,2),axis=1)/ sample
print(avg2_del_qp_particle_sample.shape)

avg_del_q_particle_sample = np.sum(avg_del_q_particle,axis=1) / sample
# print('----avg_del_q_sample_particle sum components----')
# print(avg_del_q_sample_particle)

avg_del_p_particle_sample = np.sum(avg_del_p_particle,axis=1) / sample
# print('----avg_del_p_sample_particle sum components----')
# print(avg_del_p_sample_particle)

avg_del_qp_particle_sample = avg_del_q_particle_sample + avg_del_p_particle_sample
print('----avg_del_qp_sample_particle -----')
print(avg_del_qp_particle_sample)
# print(avg_del_qp_sample_particle.shape)

std_del_qp_particle_sample = (avg2_del_qp_particle_sample - avg_del_qp_particle_sample*avg_del_qp_particle_sample)**0.5
print('----std_del_qp_particle_sample -----')
print(std_del_qp_particle_sample)

fig = plt.figure()
t = np.arange(0., iterations * time_step + time_step, time_step)
#t = np.arange(0., 3)
fig.add_subplot(2,2,1)
plt.title(r'BoxSize {:.4f}, Maximum distance $r = {:.4f}, r^2 = {:.4f}$'.format(BoxSize,q_max,q_max*q_max))
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

