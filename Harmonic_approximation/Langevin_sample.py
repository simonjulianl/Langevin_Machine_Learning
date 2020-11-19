import Langevin_Machine_Learning.hamiltonian as Hamiltonian
import Langevin_Machine_Learning.Integrator as Integrator
import Langevin_Machine_Learning.Integrator.methods as methods
import Langevin_Machine_Learning.utils as confStat # configuration statistics
import numpy as np
import argparse
import time

N_particle =2
rho=0.2
T =1
samples =1
iterations=2
ts =0.1
gamma =0
print("N_particle",N_particle)
print("rho:",rho)
print("T:",T)
print("N_samples",samples)
print("iterations",iterations)
print("ts",ts)
print("gamma",gamma)


energy = Hamiltonian.Hamiltonian()
LJ = Hamiltonian.LJ_term(epsilon =1, sigma =1, boxsize=np.sqrt(N_particle/rho)) #'density': 0.2
energy.append(Hamiltonian.Lennard_Jones(LJ, boxsize=np.sqrt(N_particle/rho))) #'density': 0.2  np.sqrt(N_particle/rho)
energy.append(Hamiltonian.kinetic_energy(mass = 1))

configuration = {
    'kB' : 1.0, # put as a constant
    'Temperature' : T, # desired temperature for NVE Ensemble
    'DIM' : 2,
    'm' : 1,
    'particle' : N_particle,
    'N' : 1,   # Total number of samples for train 50000 for test 5000
    'BoxSize': np.sqrt(N_particle/rho),  #'density' =particle/volume : 0.2 ; Boxsize : sqrt(particle/density)
    'hamiltonian' : energy,
    'R_adj' :0,
    #'pos' : np.load('Langevin_Machine_Learning/init/N{}_T{}_pos_sampled.npy'.format(2,0.01))
    }

integration_setting = {
    'iterations' : iterations,
    'DumpFreq' : 1,
    'gamma' : gamma, # gamma 0 turns off the Langevin heat bath, setting it to NVE Ensemble
    'time_step' : ts,
    }

#MD simulation
configuration.update(integration_setting)
MD_integrator = Integrator.analytic_method(**configuration)
#load for initial condition for each temperature
#change filename when test
#change filename when train
#initial_q_hist, initial_p_hist = MD_integrator.set_phase_space(samples = samples) # command out when save a file
#update configuration after loading
configuration = MD_integrator.get_configuration()
print('main',configuration)
print('Run MD simulation')
start = time.time()
MD_integrator.integrate()
quit()
end = time.time()
print('time',end-start)
print('-----------------')


data = np.stack((q_hist_ ,p_hist_), axis=0)
np.save('data_hand_gradient.npy'.format(ts),data)

configuration.update(integration_setting)

# to save the current phase space to continue as a checkpoint
#MD_integrator.save_phase_space(q_hist, p_hist,'/N{}_T{}_ts{}_iter{}_vv_gm{}_{}sampled_test.npy'.format(N_particle,T,ts,iterations,gamma,samples)) # by default, it is in init file

#Analysis
# configuration.update(integration_setting)
# MD_integrator = Integrator.Langevin(**configuration)
# initial_q_hist, initial_p_hist = MD_integrator.set_phase_space(samples = samples)
# configuration = MD_integrator.get_configuration()
# configuration.update(integration_setting)
# q_hist, p_hist = np.load('Langevin_Machine_Learning/init/N{}_T{}_ts{}_iter{}_vv_gm{}_1000sampled_predicted_1.npy'.format(N_particle,T,ts,iterations,gamma))
# print(q_hist.shape,p_hist.shape)
#
# q_hist    = q_hist.reshape(-1,samples,N_particle,configuration['DIM'])
# p_hist    = p_hist.reshape(-1,samples,N_particle,configuration['DIM'])
# print(q_hist.shape,p_hist.shape)
#
# confStat.plot_stat(q_hist, p_hist, 'all',**configuration)
# #Langevin
# confStat.plot_stat(q_hist, p_hist, 'v_dist',**configuration)
# confStat.plot_stat(q_hist, p_hist, 'instantaneous_temp',**configuration)
#plot the statistic of q distribution based on current state configuration

