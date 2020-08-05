import Langevin_Machine_Learning.hamiltonian as Hamiltonian
import Langevin_Machine_Learning.Integrator as Integrator
import Langevin_Machine_Learning.Integrator.methods as methods
import Langevin_Machine_Learning.utils as confStat # configuration statistics
import numpy as np

energy = Hamiltonian.Hamiltonian()
LJ06 = Hamiltonian.LJ_term(epsilon =1, sigma =1, exponent= 6, boxsize=np.sqrt(2/0.2)) #'density': 0.2
LJ12 = Hamiltonian.LJ_term(epsilon =1, sigma =1, exponent= 12, boxsize=np.sqrt(2/0.2)) #'density': 0.2
energy.append(Hamiltonian.Lennard_Jones(LJ06,LJ12, boxsize=np.sqrt(2/0.2))) #'density': 0.2
energy.append(Hamiltonian.kinetic_energy(mass = 1))

configuration = {
    'kB' : 1.0, # put as a constant 
    'Temperature' : 0.35, # desired temperature for NVE Ensemble
    'DIM' : 2,
    'm' : 1,
    'particle' : 2,
    'N' : 64000,   # Total number of samples
    'BoxSize': np.sqrt(2/0.2),  #'density' =particle/volume : 0.2 ; Boxsize : sqrt(particle/density)
    'periodicity' : True,
    'hamiltonian' : energy,
    'pos' : np.load('Langevin_Machine_Learning/init/N{}_T{}_pos_sampled.npy'.format(2,0.35))
    }

integration_setting = {
    'iterations' : 1000,
    'DumpFreq' : 1,
    'gamma' : 0, # gamma 0 turns off the Langevin heat bath, setting it to NVE Ensemble
    'time_step' : 0.01,
    'integrator_method' : methods.position_verlet, #method class to be passed
    }

configuration.update(integration_setting)
print(configuration)
print('\n')
print('-----------------')
print('MD_integrator')
MD_integrator = Integrator.Langevin(**configuration)
#only load for initial condition Temperature = 1.0
print('-----------------')
print('set_phase_space')
inital_q_hist, inital_p_hist = MD_integrator.set_phase_space(samples = 4) # command out when save a file
print('-----------------')
#update configuration after loading
configuration = MD_integrator.get_configuration()
#print("main.py after MD_integrator",configuration)
print('-----------------')
print('Run MD simulation')
q_hist, p_hist = MD_integrator.integrate()
#print("main.py after q_hist, p_hist ",configuration)
print('q_hist',q_hist)
print('q_hist.shape',q_hist.shape)
print('p_hist',p_hist)
print('p_hist.shape',p_hist.shape)
print('-----------------')
#confStat.kinetic_energy(**configuration)
configuration.update(integration_setting)
confStat.plot_stat(inital_q_hist, inital_p_hist,q_hist, p_hist, 'all',**configuration)

#plot the statistic of q distribution based on current state configuration

#to save the current phase space to continue as a checkpoint
MD_integrator.save_phase_space(inital_q_hist, inital_p_hist,q_hist, p_hist,'/N{}_T{}_ts{}_md_sampled.npy'.format(2,0.35,0.001)) # by default, it is in init file
