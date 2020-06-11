import Langevin_Machine_Learning.hamiltonian as Hamiltonian
import Langevin_Machine_Learning.Integrator as Integrator
import Langevin_Machine_Learning.Integrator.methods as methods 
import Langevin_Machine_Learning.utils as confStat # configuration statistics

energy = Hamiltonian.Hamiltonian()
energy.append(Hamiltonian.asymmetrical_double_well())
energy.append(Hamiltonian.kinetic_energy(mass = 1))

configuration = {
    'kB' : 1.0, # put as a constant 
    'Temperature' : 1.0, # desired temperature for NVT Ensemble
    'DIM' : 1,
    'm' : 1,
    'N' : 100,
    'hamiltonian' : energy,
    }

integration_setting = {
    'iterations' : 500,
    'DumpFreq' : 1,
    'gamma' : 0, # gamma 0 turns off the Langevin heat bath, setting it to NVE Ensemble
    'time_step' : 0.01,
    'integrator_method' : methods.position_verlet, #method class to be passed
    }

configuration.update(integration_setting)
MD_integrator = Integrator.Langevin(**configuration)
#only load for initial condition Temperature = 1.0
MD_integrator.set_phase_space(samples = 100)
    
#update configuration after loading
configuration = MD_integrator.get_configuration()
q_hist, p_hist = MD_integrator.integrate() 
confStat.plot_stat(q_hist, p_hist, 'q_dist',**configuration) 
#plot the statistic of q distribution based on current state configuration

#to save the current phase space to continue as a checkpoint
MD_integrator.save_phase_space() # by default, it is in init file
