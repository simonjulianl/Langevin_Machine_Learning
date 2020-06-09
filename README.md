# Langevin Machine Learning 


# prerequisites
```
matplotlib==3.1.1
numpy==1.17.4
torch==1.4.0
tdqm==4.43.0

python>=3.6.9
```
# code example 

1. Using Monte Carlo Markov Chain (MCMC) Integrator
```
import Langevin_Machine_Learning.hamiltonian as Hamiltonian
import Langevin_Machine_Learning.Integrator as Integrator

energy = Hamiltonian()
energy.append(asymmetrical_double_well())

configuration = {
    'kB' : 1.0, # put as a constant 
    'Temperature' : 1.0, 
    'DIM' : 1,
    'm' : 1,
    'N' : 1,
    'hamiltonian' : energy,
    }

integration_setting = {
    'iterations' : 500,
    'DumpFreq' : 1,
    'dq' : 0.05,
    }

configuration.update(integration_setting)
MSMC_integrator = Integrator.MSMC(**configuration)
q_list = MSMC_integrator.integrate()
np.save('init/p_N2500_T1_DIM1_MCMC.npy',np.array(q_list))  
```

2. Using MD Langevin Simulator ( NVT Ensemble / NVE Ensemble if \gamma = 0
