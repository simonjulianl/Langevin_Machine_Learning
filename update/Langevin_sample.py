import Langevin_Machine_Learning.hamiltonian as Hamiltonian
import Langevin_Machine_Learning.Integrator as Integrator
import Langevin_Machine_Learning.Integrator.methods as methods
import Langevin_Machine_Learning.utils as confStat # configuration statistics
import numpy as np

energy = Hamiltonian.Hamiltonian()
energy.append(Hamiltonian.Lennard_Jones(epsilon =1, sigma =1,BoxSize=3.16))
energy.append(Hamiltonian.kinetic_energy(mass = 1))

configuration = {
    'kB' : 1.0, # put as a constant 
    'Temperature' : 0.95, # desired temperature for NVE Ensemble
    'DIM' : 2,
    'm' : 1,
    'particle' : 2,
    'N' : 16000,   # Total number of samples
    'BoxSize': 3.16,
    'periodicity' : True,
    'hamiltonian' : energy,
    'pos' : np.load('Langevin_Machine_Learning/init/N{}_T{}_pos_sampled.npy'.format(2,0.95))
    }

integration_setting = {
    'iterations' : 3,
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
MD_integrator.set_phase_space(samples = 3) # command out when save a file
print('-----------------')
#update configuration after loading
configuration = MD_integrator.get_configuration()
#print(configuration)
print('-----------------')
print('Run MD simulation')
q_hist, p_hist = MD_integrator.integrate()
print('q_hist.shape',q_hist.shape)
print('p_hist.shape',p_hist.shape)
print('-----------------')
quit()
#confStat.kinetic_energy(**configuration)
confStat.plot_stat(q_hist, p_hist, 'all',**configuration)

#plot the statistic of q distribution based on current state configuration
quit()
#to save the current phase space to continue as a checkpoint
MD_integrator.save_phase_space() # by default, it is in init file
