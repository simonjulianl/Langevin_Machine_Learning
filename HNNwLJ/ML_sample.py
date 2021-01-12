import hamiltonian as hamiltonian
from hamiltonian.pb import pb
from phase_space import phase_space
import integrator as integrator
import integrator.methods as methods
import pair_wise_HNN as pair_wise_HNN
import pair_wise_HNN.models as models
from pair_wise_HNN.loss import qp_MSE_loss
import torch.optim as optim
import torch
import math

nsamples = 1
nparticle = 2
DIM = 2
mass = 1
epsilon = 1.
sigma = 1.
rho = 0.1
boxsize = math.sqrt(nparticle/rho)
tau = 0.01 # short time step for label
iterations = 10 # n times short time step to pair data at one large time step
n_input = 5
n_hidden = 5
lr = 0.01
nepochs = 50

seed = 9372211
torch.manual_seed(seed)
# torch.backends.cudnn.deterministic = True  # Processing speed may be lower then when the model functions nondeterministically.
# torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed_all(seed)

# noML_hamiltonian
noML_hamiltonian = hamiltonian.hamiltonian()
LJ = hamiltonian.LJ_term(epsilon = epsilon, sigma = sigma, boxsize = boxsize)
noML_hamiltonian.append(hamiltonian.lennard_jones(LJ, boxsize = boxsize))
noML_hamiltonian.append(hamiltonian.kinetic_energy(mass = 1))

phase_space = phase_space.phase_space()
pb = pb()

state = {
    'nsamples': nsamples,
    'nparticle': nparticle,
    'DIM': DIM,
    'boxsize': boxsize,
    'iterations': iterations,
    'tau': tau,
    'integrator_method': methods.linear_velocity_verlet,
    'phase_space': phase_space,
    'pb_q': pb,
    'nepochs' : nepochs
}

MLP = models.pair_wise_MLP(n_input, n_hidden)
opt = optim.Adam(MLP.parameters(), lr=lr)
# pair_wise_HNN = pair_wise_HNN(NoML_hamiltonian, MLP, **state)  # data preparation / calc f_MD, f_ML

setting = {
    'opt' : opt,
    'loss' : qp_MSE_loss,
    'MLP' : MLP
    }

state.update(setting)

MD_learner = pair_wise_HNN.MD_learner(integrator.linear_integrator, noML_hamiltonian, pair_wise_HNN.pair_wise_HNN)
MD_learner.train(filename ='./init_config/N_particle{}_samples{}_rho0.1_T0.04_pos_sampled.npy'.format(nparticle,nsamples), **state)
