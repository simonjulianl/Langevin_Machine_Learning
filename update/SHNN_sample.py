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
    'sample' : 2500, # samples per temperature
    }

configuration.update(integration_setting)
configuration.update(NN_trainer_setting)
configuration.update(Dataset_setting)

SHNN = HNN.SHNN_trainer(level = 2, folder_name = 'PV_Hidden20_Batch32', **configuration)
SHNN.train()
