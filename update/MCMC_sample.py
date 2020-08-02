import Langevin_Machine_Learning.hamiltonian as Hamiltonian
import Langevin_Machine_Learning.Integrator as Integrator
import Langevin_Machine_Learning.utils as confStat
import Langevin_Machine_Learning.phase_space as phase_space 

energy = Hamiltonian.Hamiltonian() # energy model container
LJ06 = Hamiltonian.LJ_term(epsilon =1, sigma =1, exponent= 6, boxsize=3.16)
LJ12 = Hamiltonian.LJ_term(epsilon =1, sigma =1, exponent= 12, boxsize=3.16)
energy.append(Hamiltonian.Lennard_Jones(LJ06,LJ12))

configuration = {
    'kB' : 1.0, # put as a constant 
    'Temperature' : 0.95,
    'DIM' : 2,
    'm' : 1,
    'particle' : 2,
    'N' : 5,
    'BoxSize': 3.16,
    'periodicity': True,
    'hamiltonian' : energy,
    }

integration_setting = {
    'iterations' : 5,
    'DumpFreq' : 1,
    'dq' : 0.01,
    }

configuration.update(integration_setting) # combine the 2 dictionaries

MSMC_integrator = Integrator.MCMC(**configuration)
q_hist = MSMC_integrator.integrate()

# total samples used in momentum sampler is iterations / dumpfreq as they are usually used together 
# Momentum_sampler = Integrator.msiomentum_sampler(**configuration)
# p_hist = Momentum_sampler.integrate()
# print(p_hist.shape)
# p_hist = p_hist.reshape(q_hist.shape)
# print(p_hist.shape)
# confStat.plot_stat(q_hist, p_hist, 'q_dist', **configuration)
quit()
DIM = q_hist.shape[-1] # flatten the q_hist of samples x N X DIM to a phase space
q_list = q_hist.reshape(-1,DIM)
# p_list = p_hist.reshape(-1,DIM)
phase_space = phase_space.phase_space() # wrapper of phase space class
phase_space.set_q(q_list)
# phase_space.set_p(p_list)
phase_space.write(filename = "Particle2_T{}_pos_sampled.npy".format(0.95))
#the folder init could be used for next NN / MD initialization, check the convention of name saving
