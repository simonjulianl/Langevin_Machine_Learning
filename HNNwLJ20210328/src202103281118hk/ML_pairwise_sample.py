import sys
import os
sys.path.append(os.path.abspath("./parameters"))

from HNN                        import pair_wise_HNN
from HNN.models                 import pair_wise_MLP
# from HNN.models               import pair_wise_zero
from HNN.MD_learner             import MD_learner
from HNN.dataset                import dataset
from HNN.MD_tester              import MD_tester
from parameters.MC_parameters   import MC_parameters
from parameters.MD_parameters   import MD_parameters
from parameters.ML_parameters   import ML_parameters
from phase_space                import phase_space
from integrator                 import linear_integrator
from utils.data_io              import data_io
import torch
import time

def qp_list_combine(init_filename, mode):

    qp_list_app = []
    for i, temp in enumerate(MD_parameters.temp_list):
        # qp_list shape is [(q,p), nsamples, nparticle, DIM]
        qp_list = data_io_obj.read_init_qp(init_filename + 'T{}_pos_'.format(temp) + str(mode) + '_sampled.pt')
        qp_list_app.append(qp_list)

    return qp_list_app

def phase_space_mode(qp_list, phase_space):

    phase_space.set_q(qp_list[0])
    phase_space.set_p(qp_list[1])

    return phase_space

nsamples = MD_parameters.nsamples
nparticle = MC_parameters.nparticle
temp = MD_parameters.temp_list
rho = MC_parameters.rho
tau_long = MD_parameters.tau_long
tau_short = MD_parameters.tau_short
append_strike = MD_parameters.append_strike
niter_tau_long = MD_parameters.niter_tau_long
niter_tau_short = MD_parameters.niter_tau_short

lr = ML_parameters.lr
optname = ML_parameters.opt.name()
MLP_nhidden = ML_parameters.MLP_nhidden
activation = ML_parameters.activation
crash_duplicate_ratio = MD_parameters.crash_duplicate_ratio

print('nparticle tau_long tau_short lr nsamples_batch MLP_nhidden')
print(nparticle, tau_long, tau_short, lr, MD_parameters.nsamples_batch, ML_parameters.MLP_nhidden )

seed = ML_parameters.seed
torch.manual_seed(seed)
# torch.backends.cudnn.deterministic = True  # Processing speed may be lower then when the models functions nondeterministically.
# torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed_all(seed)

phase_space = phase_space.phase_space()
linear_integrator_obj = linear_integrator( MD_parameters.integrator_method )
pair_wise_HNN_obj = pair_wise_HNN(pair_wise_MLP().to(ML_parameters.device))
noMLhamiltonian = super(type(pair_wise_HNN_obj), pair_wise_HNN_obj)


root_train_path = './saved_model/'
root_retrain_path = './retrain_saved_model/'

prefix = 'nsamples{}_nparticle{}_tau{}_{}_lr{}_h{}_{}'.format( nsamples, nparticle, tau_long, optname, lr, MLP_nhidden, activation)
load_path = root_train_path + prefix + '_checkpoint.pth'

# path for train
best_model_path = root_train_path + prefix + '_checkpoint_best.pth'
save_path = root_train_path + prefix + '_checkpoint.pth'
write_loss_filename = prefix + '_loss.txt'

# # path for retrain
# best_model_path = root_retrain_path + prefix + '_crash_{}_checkpoint_best.pth'.format(crash_duplicate_ratio)
# save_path = root_retrain_path + prefix + '_crash_{}_checkpoint.pth'.format(crash_duplicate_ratio)
# write_loss_filename = prefix + '_crash_{}_loss.txt'.format(crash_duplicate_ratio)

uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])
base_dir = uppath(__file__, 2)
init_path = base_dir + '../data/'
filename = '../data/tmp/nparticle{}_tau{}'.format(nparticle, tau_short)

torch.autograd.set_detect_anomaly(True)

# for train
data_io_obj = data_io(init_path)
dataset_obj = dataset()

init_filename = 'init_config/nparticle{}_new_nsim_rho{}_'.format(nparticle, rho)

print("============ train data ===============")
qp_train_list = qp_list_combine(init_filename, 'train')
qp_train_list = torch.cat(qp_train_list, dim=1)
qp_train = dataset_obj.qp_dataset(qp_train_list)

print("============ valid data ===============")
qp_valid_list = qp_list_combine(init_filename, 'valid')
qp_valid_list = torch.cat(qp_valid_list, dim=1)
qp_valid = dataset_obj.qp_dataset(qp_valid_list)

print("============ label for train ===============")
phase_space = phase_space_mode(qp_train, phase_space)
qp_train_label = linear_integrator_obj.nsteps(noMLhamiltonian, phase_space, tau_short, niter_tau_short, append_strike)
qp_train_label = torch.stack(qp_train_label)

print("============ label for valid ===============")
phase_space = phase_space_mode(qp_valid, phase_space)
qp_valid_label = linear_integrator_obj.nsteps(noMLhamiltonian, phase_space, tau_short, niter_tau_short, append_strike)
qp_valid_label = torch.stack(qp_valid_label)

dataset = [qp_train, qp_valid, qp_train_label, qp_valid_label]

MD_learner = MD_learner(linear_integrator_obj, pair_wise_HNN_obj, phase_space, dataset, load_path)


