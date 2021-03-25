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
import time

nsamples = MD_parameters.nsamples
nparticle = MC_parameters.nparticle
temp = MD_parameters.temp_list
tau_long = MD_parameters.tau_long
tau_short = MD_parameters.tau_short
nsamples_cur = nsamples
tau_cur = tau_short # find gold standard
pair_interval = int(tau_long / tau_short)
MD_iterations = round(MD_parameters.max_ts / tau_short)
iteration_batch = MD_parameters.iteration_batch
iteration_save_batch = iteration_batch * int(MD_parameters.tau_long * tau_cur)

print('nparticle tau pair_interval MD_iterations')
print(nparticle, tau_short, pair_interval, MD_iterations)

seed = MD_parameters.seed
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

if not os.path.exists('./tmp/'):
    os.makedirs('./tmp/')

uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])
base_dir = uppath(__file__, 1)
init_test_path = base_dir + '/init_config_for_testset/'
filename = 'tmp/nparticle{}_tau{}'.format(nparticle, tau_cur)
base_library = os.path.abspath('gold_standard')

data_io_obj = data_io(init_test_path)
init_q, init_p = data_io_obj.loadq_p('test')

# print('init', init_q, init_p)

phase_space.set_q(init_q)
phase_space.set_p(init_p)

start = time.time()
linear_integrator_obj.step( noMLhamiltonian, phase_space, MD_iterations, nsamples_cur, tau_cur)
end = time.time()
print('time for integrator :', end-start)
print('start concat')
start = time.time()
q_list, p_list = linear_integrator_obj.concat_step(MD_iterations, tau_cur)
end = time.time()
print('time for save files :', end-start)

print('end concat')
print(q_list.shape, p_list.shape)
print('time for concat :', end-start)

init_q = torch.unsqueeze(init_q, dim=0)
init_p = torch.unsqueeze(init_p, dim=0)

q_hist = torch.cat((init_q, q_list), dim=0)
p_hist = torch.cat((init_p, p_list), dim=0)

# pair w large time step

q_hist = torch.unsqueeze(q_hist, dim=1)
p_hist = torch.unsqueeze(p_hist, dim=1)

qp_hist = torch.cat((q_hist, p_hist), dim=1)

torch.save(qp_hist, base_library + '/nparticle{}_T{}_ts{}_iter{}_vv_{}sampled.pt'.format(nparticle,temp[0],tau_short,MD_iterations,nsamples))

# remove files
for z in range(int(MD_iterations / iteration_save_batch)):
    os.remove( filename + '_{}.pt'.format(z))
