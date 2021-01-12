import Langevin_Machine_Learning.hamiltonian as Hamiltonian
import Langevin_Machine_Learning.Integrator as Integrator
import time
import numpy as np

energy = Hamiltonian.Hamiltonian()
LJ = Hamiltonian.LJ_term(epsilon =1, sigma =1, boxsize=8) #'density': 0.2
energy.append(Hamiltonian.Lennard_Jones(LJ, boxsize=8)) #'density': 0.2  np.sqrt(N_particle/rho)
energy.append(Hamiltonian.kinetic_energy(mass = 1))

T=0.00001
N_particle=4

configuration = {
    'kB' : 1.0, # put as a constant
    'Temperature' : T, # desired temperature for NVE Ensemble
    'DIM' : 2,
    'm' : 1,
    'particle' : N_particle,
    'N' : 1,   # Total number of samples for train 50000 for test 5000
    'BoxSize': 8,  #'density' =particle/volume : 0.2 ; Boxsize : sqrt(particle/density)
    'hamiltonian' : energy,
    'R_adj' :0
    }


#MD simulation
MD_integrator = Integrator.Finite_difference_first_derivative(**configuration)
start = time.time()
configuration = MD_integrator.get_configuration()
print('Run MD simulation')
first_derivative = MD_integrator.integrate()
end = time.time()
print('time',end-start)
print('derivative_U_q1x',first_derivative[0])
print('derivative_U_q1y',first_derivative[1])
print('derivative_U_q2x',first_derivative[2])
print('derivative_U_q2y',first_derivative[3])
print('derivative_U_q3x',first_derivative[4])
print('derivative_U_q3y',first_derivative[5])
print('derivative_U_q4x',first_derivative[6])
print('derivative_U_q4y',first_derivative[7])
print('-----------------')
MD_integrator_2 = Integrator.Finite_difference_second_derivative(**configuration)
second_derivative = MD_integrator_2.integrate()
print(second_derivative.shape)
second_derivative = second_derivative[:,0]
print('-----------------')


