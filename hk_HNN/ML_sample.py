import hamiltonian as NoML_hamiltonian
from LJ_term import LJ_term
from lennard_jones import lennard_jones
from kinetic_energy import kinetic_energy
from pb import pb
from phase_space import phase_space
from linear_integrator import linear_integrator
from linear_velocity_verlet import linear_velocity_verlet
from pair_wise_HNN import pair_wise_HNN
from pair_wise_MLP import pair_wise_MLP
from MD_learner import MD_learner
from loss import qp_MSE_loss
import torch.optim as optim

nsamples = 1
N_particle = 2
DIM = 2
mass = 1
epsilon = 1.
sigma = 1.
boxsize = 6.  # np.sqrt(N_particle/rho)
tau = 0.01
iterations = 10
n_input = 5
n_hidden = 5
lr = 0.01
nepochs = 1


# noML_hamiltonian
noML_hamiltonian = NoML_hamiltonian.hamiltonian()
LJ = LJ_term(epsilon = 1, sigma = 1, boxsize = boxsize)
noML_hamiltonian.append(lennard_jones(LJ, boxsize = boxsize))
noML_hamiltonian.append(kinetic_energy(mass = 1))

phase_space = phase_space()
pb = pb()

state = {
    'N': nsamples,
    'particle': N_particle,
    'DIM': DIM,
    'BoxSize': boxsize,
    'iterations': iterations,
    'tau': tau,
    'integrator_method': linear_velocity_verlet,
    'phase_space': phase_space,
    'pb_q': pb,
    'nepochs' : nepochs
}

MLP = pair_wise_MLP(n_input, n_hidden)
opt = optim.Adam(MLP.parameters(), lr=lr)
pairwise_HNN = pair_wise_HNN(NoML_hamiltonian, MLP, **state)  # data preparation / calc f_MD, f_ML

setting = {
    'opt' : opt,
    'loss' : qp_MSE_loss
    }

state.update(setting)

MD_learner = MD_learner(linear_integrator, noML_hamiltonian, pairwise_HNN)
MD_learner.train(filename ='N_particle2_samples1_rho0.1_T0.04_pos_sampled.npy', **state)
