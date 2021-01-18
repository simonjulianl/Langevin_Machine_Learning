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

nsamples_label = 50
nsamples_ML = 1
nparticle = 2
DIM = 2
mass = 1
epsilon = 1.
sigma = 1.
rho = 0.1
boxsize = math.sqrt(nparticle/rho)
tau_short = 0.01 # short time step for label
tau_long = 0.01
n_input = 5
n_hidden = 40
lr = 0.0001
nepochs = 100000

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
pb = pb()     # pb is boundary condition

state = {
    'nsamples_cur': 0,
    'nsamples_label': nsamples_label,
    'nsamples_ML': nsamples_ML,
    'nparticle': nparticle,
    'DIM': DIM,
    'boxsize': boxsize,
    'MD_iterations': 0,
    'tau_cur': 0.0,
    'tau_long' : tau_long,  # for MD
    'tau_short' : tau_short,# for label (gold standard)
    'integrator_method': methods.linear_velocity_verlet,
    'phase_space': phase_space,
    'pb_q': pb,
    'nepochs' : nepochs
}

MLP = models.pair_wise_MLP(n_input, n_hidden)
opt = optim.Adam(MLP.parameters(), lr=lr)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

setting = {
    'opt' : opt,
    'loss' : qp_MSE_loss,
    'MLP' : MLP,
    '_device' : device
    }

state.update(setting)

# check gpu available
print('__Number CUDA Devices:', torch.cuda.device_count())
print('__Devices')
print('Active CUDA Device: GPU', torch.cuda.current_device())
print ('Available devices ', torch.cuda.device_count())
print ('Current cuda device ', torch.cuda.current_device())
print('GPU available', torch.cuda.get_device_name(device))

MD_learner = pair_wise_HNN.MD_learner(integrator.linear_integrator, noML_hamiltonian, pair_wise_HNN.pair_wise_HNN)
MD_learner.trainer(filename ='./init_config/N_particle{}_samples{}_rho0.1_T0.04_pos_sampled.pt'.format(nparticle, nsamples_label), **state)
