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
temp = MC_parameters.temperature
tau_long = MD_parameters.tau_long
tau_short = MD_parameters.tau_short
nsamples_cur = nsamples
tau_cur = tau_short # find gold standard
pair_interval = int(tau_long / tau_short)
MD_iterations = round(MD_parameters.max_ts / tau_short)

print('nparticle tau pair_interval MD_iterations')
print(nparticle, tau_short, pair_interval, MD_iterations)

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

phase_space.set_q(init_q)
phase_space.set_p(init_p)

q_list, p_list = linear_integrator_obj.step( noMLhamiltonian, phase_space, MD_iterations, nsamples_cur, tau_cur)

init_q = torch.unsqueeze(init_q, dim=0)
init_p = torch.unsqueeze(init_p, dim=0)

q_hist_ = torch.cat((init_q, q_list), dim=0)
p_hist_ = torch.cat((init_p, p_list), dim=0)

# pair w large time step
q_hist = q_hist_[0::pair_interval]
p_hist = p_hist_[0::pair_interval]

base_library = os.path.abspath('gold_standard')

torch.save((q_hist, p_hist), base_library + '/nsample{}_T{}_ts{}_iter{}_vv_{}sampled.pt'.format(nparticle,temp,tau_short,MD_iterations,nsamples))