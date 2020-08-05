import numpy as np

gold_standard = np.load('Langevin_Machine_Learning/init/N2_T0.35_ts0.001_md_sampled.npy')
tau_01 = np.load('Langevin_Machine_Learning/init/N2_T0.35_ts0.01_md_sampled.npy')

print(gold_standard.shape)
print(tau_01.shape)

gs_q = gold_standard[0]
q    = tau_01[0]

print(gs_q[:,0].shape)
print(gs_q[0,1])
print(q[:,0].shape)
print(q[0,1])



