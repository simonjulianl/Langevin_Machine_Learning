import Langevin_Machine_Learning.hamiltonian as Hamiltonian
import Langevin_Machine_Learning.Integrator as Integrator
import Langevin_Machine_Learning.utils as confStat

energy = Hamiltonian.Hamiltonian() # energy model container
energy.append(Hamiltonian.asymmetrical_double_well())

configuration = {
    'kB' : 1.0, # put as a constant 
    'Temperature' : 1.0, 
    'DIM' : 1,
    'm' : 1,
    'N' : 1,
    'hamiltonian' : energy,
    }

integration_setting = {
    'iterations' : 50000,
    'DumpFreq' : 1,
    'dq' : 1.0,
    }

configuration.update(integration_setting) # combine the 2 dictionaries

MSMC_integrator = Integrator.MSMC(**configuration)
q_list = MSMC_integrator.integrate()
p_list_dummy = np.zeros(q_list.shape) # to be passed to confstat
confStat.plot_stat(q_list, p_list_dummy, 'q_dist', **configuration)
np.save('init/p_N2500_T1_DIM1_MCMC.npy',np.array(q_list))  
#the folder init could be used for next NN / MD initialization 
