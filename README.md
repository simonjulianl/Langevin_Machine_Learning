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
Full documentation could be found by using python ```help()``` function on the enquired class 

1. Using Monte Carlo Markov Chain (MCMC) Integrator
```
import Langevin_Machine_Learning.hamiltonian as Hamiltonian
import Langevin_Machine_Learning.Integrator as Integrator

energy = Hamiltonian() # energy model container
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

configuration.update(integration_setting) # combine the 2 dictionaries

MSMC_integrator = Integrator.MSMC(**configuration)
q_list = MSMC_integrator.integrate()
np.save('init/p_N2500_T1_DIM1_MCMC.npy',np.array(q_list))  
#the folder init could be used for next NN / MD initialization 
```

2. Using MD Langevin Simulator ( NVT Ensemble / NVE Ensemble if gamma = 0 )
```
import Langevin_Machine_Learning.hamiltonian as Hamiltonian
import Langevin_Machine_Learning.Integrator as Integrator
import Langevin_Machine_Learning.Integrator.methods as methods 
import Langevin_Machine_Learning.utils.confStat as confStat

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
q_list, p_list = MD_integrator.integrate() 
confStat.plot_stat(q_list, p_list, 'q_dist',**configuration) 
#plot the statistic of q distribution based on current state configuration
```
3. Using Stacked NN trainer
```
from Langevin_Machine_Learning.HNN.loss import qp_MSE_loss  
import Langevin_Machine_Learning.HNN as HNN
import Langevin_Machine_Learning.hamiltonian as Hamiltonian
import Langevin_Machine_Learning.Integrator.methods as methods 
import torch.optim as optim 

energy = Hamiltonian.Hamiltonian() # base constructor container
energy.append(Hamiltonian.asymmetrical_double_well())
energy.append(Hamiltonian.kinetic_energy(mass = 1))

configuration = {
    'kB' : 1.0, # put as a constant 
    'Temperature' : 1.0, 
    'DIM' : 1,
    'm' : 1,
    'hamiltonian' : energy,
    }

#Generate dataset ground truth, turn off Langevin drag coefficient
integration_setting = {
    'iterations' : 500,
    'DumpFreq' : 1,
    'gamma' : 0, # turn off Langevin heat bath
    'time_step' : 0.001,
    'integrator_method' : methods.position_verlet
    }
    
model = HNN.MLP2H_Separable_Hamil_PV(2, 20)
loss = qp_MSE_loss

lr = 1e-3
NN_trainer_setting = {
    'optim' : optim.Adam(model.parameters(), lr = lr),
    'model' : model,
    'loss' : loss,
    'epoch' : 500, 
    'batch_size' : 32,
    }


Dataset_setting = {
    'Temperature_List' : [1,2,3,4,5,6,7,8,9,10],
    'sample' : 2500,
    }

configuration.update(integration_setting)
configuration.update(NN_trainer_setting)
configuration.update(Dataset_setting)

SHNN = HNN.SHNN_trainer(level = 2, folder_name = 'PV_Hidden20_Batch32', **configuration)
SHNN.train()
```
