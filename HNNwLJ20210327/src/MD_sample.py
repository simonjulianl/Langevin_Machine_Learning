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
from utils.data_io import data_io
import torch
import time

nsamples = MD_parameters.nsamples
nparticle = MC_parameters.nparticle
temp = MD_parameters.temp_list
tau_long = MD_parameters.tau_long
tau_short = MD_parameters.tau_short
nsamples_cur = nsamples
tau_cur = tau_short # find gold standard
paird_step = int(tau_long / tau_short)
MD_iterations = round(MD_parameters.max_ts / tau_short)
iteration_batch = MD_parameters.iteration_batch
iteration_save_batch = int( paird_step * iteration_batch)
nfile = int(MD_iterations / iteration_save_batch )

print('nparticle tau pair_interval MD_iterations')
print(nparticle, tau_short, paird_step, MD_iterations)

# seed setting
seed = MD_parameters.seed
torch.manual_seed(seed)
# torch.backends.cudnn.deterministic = True  # Processing speed may be lower then when the models functions nondeterministically.
# torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed_all(seed)

uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])
base_dir = uppath(__file__, 2)

init_path =  base_dir + '/data/init_config/'
write_path =  base_dir + '/data/gold_standard/'
tmp_path =  base_dir + '/data/tmp/'

if not os.path.exists(write_path):
                os.makedirs(write_path)

if not os.path.exists(tmp_path):
    os.makedirs(tmp_path)

tmp_filename = tmp_path + 'nparticle{}_T{}_tau{}'.format(nparticle, temp[0], tau_cur)
filename = write_path + '/nparticle{}_T{}_ts{}_iter{}_vv_{}sampled'.format(nparticle,temp[0],tau_short,MD_iterations,nsamples)

data_io_obj = data_io(init_path)
phase_space = phase_space.phase_space()
linear_integrator_obj = linear_integrator( MD_parameters.integrator_method )
pair_wise_HNN_obj = pair_wise_HNN(pair_wise_MLP())

noMLhamiltonian = super(type(pair_wise_HNN_obj), pair_wise_HNN_obj)

init_q, init_p = data_io_obj.read_init_qp('test')

phase_space.set_q(init_q)
phase_space.set_p(init_p)

# iteration_save_batch : iterations for save files
for i in range(1, nfile):

    qp_list = linear_integrator_obj.step( noMLhamiltonian, phase_space, iteration_save_batch,  tau_cur)
    tensor_qp_list = torch.stack(qp_list)
    tensor_qp_list = tensor_qp_list[0::paird_step]

    data_io_obj.write_trajectory_qp(tmp_filename, i, tensor_qp_list)

    tensor_q = tensor_qp_list[-1,0]
    tensor_p = tensor_qp_list[-1,1]

    phase_space.set_q(tensor_q)
    phase_space.set_p(tensor_p)

qp_list = []
for i in range(1, nfile):

    # shape is [iteration_batch, (q, p), nsamples, nparticle, DIM]
    qp = data_io_obj.read_trajectory_qp(tmp_filename, i)
    qp_list.append(qp)

tensor_qp_list = torch.cat(qp_list)

q_list = tensor_qp_list[:,0]
p_list = tensor_qp_list[:,1]

init_q = torch.unsqueeze(init_q, dim=0)
init_p = torch.unsqueeze(init_p, dim=0)

q_hist = torch.cat((init_q, q_list), dim=0)
p_hist = torch.cat((init_p, p_list), dim=0)

q_hist = torch.unsqueeze(q_hist, dim=1)
p_hist = torch.unsqueeze(p_hist, dim=1)

qp_hist = torch.cat((q_hist, p_hist), dim=1)

data_io_obj.write_trajectory_qp(filename, 0, qp_hist)

# remove files
for z in range(1,nfile):
    os.remove( tmp_filename + '_{}.pt'.format(z))
