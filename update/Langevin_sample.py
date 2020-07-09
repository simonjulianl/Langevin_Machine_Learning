import Langevin_Machine_Learning.hamiltonian as Hamiltonian
import Langevin_Machine_Learning.Integrator as Integrator
import Langevin_Machine_Learning.Integrator.methods as methods 
import Langevin_Machine_Learning.utils as confStat # configuration statistics
import numpy as np

print('1')
energy = Hamiltonian.Hamiltonian()
print('2')
energy.append(Hamiltonian.Lennard_Jones(epsilon =1, sigma =1))
print('3')
energy.append(Hamiltonian.kinetic_energy(mass = 1))
print('energy \n',energy)

configuration = {
    'kB' : 1.0, # put as a constant 
    'Temperature' : 0.95, # desired temperature for NVT Ensemble
    'DIM' : 2,
    'm' : 1,
    'particle' : 2,
    'N' : 64000,   # Total number of particle
    'BoxSize': 0.55,
    'periodicity' : True,
    'hamiltonian' : energy,
    'pos' : np.load('Langevin_Machine_Learning/init/N{}_T{}_pos_sampled.npy'.format(2,0.55))
    }

integration_setting = {
    'iterations' : 10,
    'DumpFreq' : 1,
    'gamma' : 0, # gamma 0 turns off the Langevin heat bath, setting it to NVE Ensemble
    'time_step' : 0.01,
    'integrator_method' : methods.position_verlet, #method class to be passed
    }

configuration.update(integration_setting)
print(configuration)
print('\n')
print('MD_integrator')
MD_integrator = Integrator.Langevin(**configuration)
print("-----------------")
print('MD_integrator: ',MD_integrator)
print("-----------------")
#only load for initial condition Temperature = 1.0
MD_integrator.set_phase_space(samples = 4) # command out when save a file

#update configuration after loading
configuration = MD_integrator.get_configuration()
print('\n')
print(configuration)
print('\n')
q_hist, p_hist = MD_integrator.integrate()
print('q_hist.shape',q_hist.shape)
print('p_hist.shape',p_hist.shape)

confStat.plot_stat(q_hist, p_hist, 'all',**configuration)
quit()
#plot the statistic of q distribution based on current state configuration

#to save the current phase space to continue as a checkpoint
MD_integrator.save_phase_space() # by default, it is in init file
