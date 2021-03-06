import Langevin_Machine_Learning.hamiltonian as Hamiltonian
import Langevin_Machine_Learning.Integrator as Integrator
import Langevin_Machine_Learning.utils as confStat
import Langevin_Machine_Learning.phase_space as phase_space 

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
    'iterations' : 2500,
    'DumpFreq' : 1,
    'dq' : 1.0,
    }

configuration.update(integration_setting) # combine the 2 dictionaries

MSMC_integrator = Integrator.MSMC(**configuration)
q_hist = MSMC_integrator.integrate()

# total samples used in momentum sampler is iterations / dumpfreq as they are usually used together 
Momentum_sampler = Integrator.momentum_sampler(**configuration)
p_hist = Momentum_sampler.integrate()
p_hist = p_hist.reshape(q_hist.shape)
confStat.plot_stat(q_hist, p_hist, 'q_dist', **configuration)

DIM = q_hist.shape[-1] # flatten the q_hist of samples x N X DIM to a phase space
q_list = q_hist.reshape(-1,DIM)
p_list = p_hist.reshape(-1,DIM)
phase_space = phase_space.phase_space() # wrapper of phase space class
phase_space.set_q(q_list)
phase_space.set_p(p_list)
phase_space.write(filename = ./PATH TO INIT FOLDER/phase_space_N2500_T1_DIM1.npy) 
#the folder init could be used for next NN / MD initialization, check the convention of name saving
