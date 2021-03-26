import sys
import os
sys.path.append(os.path.abspath("./parameters"))

from HNN import pair_wise_HNN
from HNN.models import pair_wise_MLP
# from HNN.models import pair_wise_zero
from HNN.MD_learner import MD_learner
from HNN.MD_tester import MD_tester
from parameters.MC_parameters import MC_parameters
from parameters.MD_parameters import MD_parameters
from parameters.ML_parameters import ML_parameters
from phase_space import phase_space
from integrator import linear_integrator
import torch
import time

nsamples = MD_parameters.nsamples
nparticle = MC_parameters.nparticle
temp = MD_parameters.temp_list
rho = MC_parameters.rho
tau_long = MD_parameters.tau_long
tau_short = MD_parameters.tau_short
lr = ML_parameters.lr
optname = ML_parameters.opt.name()
MLP_nhidden = ML_parameters.MLP_nhidden
activation = ML_parameters.activation
crash_duplicate_ratio = MD_parameters.crash_duplicate_ratio
MD_iterations = round(MD_parameters.tau_long / tau_short)
iteration_batch = MD_parameters.iteration_batch
iteration_pair_batch = iteration_batch * int(MD_parameters.tau_long / tau_short)

print('nparticle tau_long tau_short lr nsamples_batch MLP_nhidden')
print(nparticle, tau_long, tau_short, lr, MD_parameters.nsamples_batch, ML_parameters.MLP_nhidden )

seed = MD_parameters.seed
torch.manual_seed(seed)
# torch.backends.cudnn.deterministic = True  # Processing speed may be lower then when the models functions nondeterministically.
# torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed_all(seed)

phase_space = phase_space.phase_space()
linear_integrator_obj = linear_integrator( MD_parameters.integrator_method )
pair_wise_HNN_obj = pair_wise_HNN(pair_wise_MLP().to(ML_parameters.device))

if not os.path.exists('./tmp/'):
    os.makedirs('./tmp/')

root_train_path = './saved_model/'
root_retrain_path = './retrain_saved_model/'

prefix = 'nsamples{}_nparticle{}_tau{}_{}_lr{}_h{}_{}'.format( nsamples, nparticle, tau_long, optname, lr, MLP_nhidden, activation)
load_path = root_train_path + prefix + '_checkpoint.pth'

# path for train
best_model_path = root_train_path + prefix + '_checkpoint_best.pth'
save_path = root_train_path + prefix + '_checkpoint.pth'
loss_curve = prefix + '_loss.txt'

# # path for retrain
# best_model_path = root_retrain_path + prefix + '_crash_{}_checkpoint_best.pth'.format(crash_duplicate_ratio)
# save_path = root_retrain_path + prefix + '_crash_{}_checkpoint.pth'.format(crash_duplicate_ratio)
# loss_curve = prefix + '_crash_{}_loss.txt'.format(crash_duplicate_ratio)

uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])
base_dir = uppath(__file__, 1)
init_path = base_dir + '/init_config/'
init_test_path = base_dir + '/init_config_for_testset/'
filename = 'tmp/nparticle{}_tau{}'.format(nparticle, tau_short)

torch.autograd.set_detect_anomaly(True)

# for train
MD_learner = MD_learner(linear_integrator_obj, pair_wise_HNN_obj, phase_space, init_path, load_path)
MD_learner.train_valid_epoch(save_path, best_model_path, loss_curve)
#
# # remove files
# for z in range(int(MD_iterations / iteration_pair_batch)):
#     os.remove( filename + '_{}.pt'.format(z))

# # for test
# start = time.time()
# MD_tester = MD_tester(linear_integrator_obj, pair_wise_HNN_obj, phase_space, init_test_path, load_path)
# q_crash_before_pred_, p_crash_before_pred_ = MD_tester.step(filename)
# end = time.time()
# print('q or p crash before pred', len(q_crash_before_pred_), len(p_crash_before_pred_))
# print('test time:', end - start)
#
# if len(q_crash_before_pred_) != 0 and len(p_crash_before_pred_) != 0 :
#
#     q_crash_before_pred = torch.unsqueeze(q_crash_before_pred_, dim=0)
#     p_crash_before_pred = torch.unsqueeze(p_crash_before_pred_, dim=0)
#
#     qp_crash_before_pred = torch.cat((q_crash_before_pred, p_crash_before_pred), dim=0)
#
#     torch.save(qp_crash_before_pred, init_path + '/nparticle{}_new_nsim_rho{}_T{}_pos_test_before_crash_sampled.pt'.format(nparticle,rho,temp[0]))

# # for crash relearner
# MD_relearner = MD_learner(linear_integrator_obj, pair_wise_HNN_obj, phase_space, init_path, load_path, crash_filename = 'test_before_crash')
# MD_relearner.train_valid_epoch(save_path, best_model_path, loss_curve)

