import sys
import os
sys.path.append(os.path.abspath("./parameters"))

from HNN import pair_wise_HNN
from HNN.models import pair_wise_MLP
# from HNN.models import pair_wise_zero
from HNN.MD_learner import MD_learner
from parameters.MC_parameters import MC_parameters
from parameters.MD_parameters import MD_parameters
from phase_space import phase_space
from integrator import linear_integrator
from HNN.data_io import data_io
import torch

nsamples = MD_parameters.nsamples
nparticle = MC_parameters.nparticle
tau_long = MD_parameters.tau_long
tau_short = MD_parameters.tau_short
nsamples_cur = nsamples
tau_cur = tau_long
MD_iterations = int(tau_long / tau_short)

print('nparticle tau_long tau_short  nsamples_batch')
print(nparticle, tau_long, tau_short, MD_parameters.nsamples_batch)

seed = 9372211
torch.manual_seed(seed)
# torch.backends.cudnn.deterministic = True  # Processing speed may be lower then when the models functions nondeterministically.
# torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed_all(seed)

phase_space = phase_space.phase_space()
linear_integrator_obj = linear_integrator( MD_parameters.integrator_method )
pair_wise_HNN_obj = pair_wise_HNN(pair_wise_MLP())

noMLhamiltonian = super(type(pair_wise_HNN_obj), pair_wise_HNN_obj)

if not os.path.exists('./gold_standard/'):
                os.makedirs('./gold_standard/')

uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])
base_dir = uppath(__file__, 1)
init_path = base_dir + '/init_config/'

data_io_obj = data_io(init_path)
init_q, init_p = data_io_obj.loadq_p('valid')
print(init_q, init_p)
phase_space.set_q(init_q)
phase_space.set_p(init_p)

q_hist, p_hist = linear_integrator_obj.step( noMLhamiltonian, phase_space, MD_iterations, nsamples_cur, tau_cur)
print(q_hist[-1])
print(p_hist[-1])

