import sys
import os
sys.path.append(os.path.abspath("./parameters"))

from HNN import pair_wise_HNN
from HNN.models import pair_wise_MLP
# from HNN.models import pair_wise_zero
from HNN.MD_learner import MD_learner
from HNN.MD_tester import MD_tester
from HNN.MD_crash_relearner import MD_crash_relearner
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
optimizer = ML_parameters.optimizer
MLP_nhidden = ML_parameters.MLP_nhidden
activation = ML_parameters.activation
crash_duplicate_ratio = MD_parameters.crash_duplicate_ratio
ML_iterations = int(MD_parameters.max_ts / MD_parameters.tau_long)

print('nparticle tau_long tau_short lr nsamples_batch MLP_nhidden')
print(nparticle, tau_long, tau_short, lr, MD_parameters.nsamples_batch, ML_parameters.MLP_nhidden )

seed = 9372211
torch.manual_seed(seed)
# torch.backends.cudnn.deterministic = True  # Processing speed may be lower then when the models functions nondeterministically.
# torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed_all(seed)

phase_space = phase_space.phase_space()
linear_integrator_obj = linear_integrator( MD_parameters.integrator_method )
pair_wise_HNN_obj = pair_wise_HNN(pair_wise_MLP().to(ML_parameters.device))

# check gpu available
# print('__Number CUDA Devices:', torch.cuda.device_count())
# print('__Devices')
# print('Active CUDA Device: GPU', torch.cuda.current_device())
# print ('Available devices ', torch.cuda.device_count())
# print ('Current cuda device ', torch.cuda.current_device())
# print('GPU available', torch.cuda.get_device_name(device))

load_path = './saved_model/nsamples{}_nparticle{}_tau{}_{}_lr{}_h{}_{}_checkpoint.pth'.format( nsamples, nparticle, tau_long, optimizer,
                                                 lr, MLP_nhidden, activation)

# # train
# best_model_path = './saved_model/nsamples{}_nparticle{}_tau{}_{}_lr{}_h{}_{}_checkpoint_best.pth'.format( nsamples, nparticle, tau_long, optimizer,
#                                                      lr, MLP_nhidden, activation)
# save_path = './saved_model/nsamples{}_nparticle{}_tau{}_{}_lr{}_h{}_{}_checkpoint.pth'.format( nsamples, nparticle, tau_long, optimizer,
#                                                  lr, MLP_nhidden, activation)
# loss_curve = 'nsamples{}_nparticle{}_tau{}_{}_{}_lr{}_h{}_{}_loss.txt'.format(nsamples, nparticle,  tau_long, tau_short, optimizer, lr, MLP_nhidden, activation)


# when retrain
best_model_path = './retrain_saved_model/nsamples{}_nparticle{}_tau{}_{}_lr{}_h{}_{}_crash_{}_checkpoint_best.pth'.format( nsamples, nparticle, tau_long, optimizer,
                                                     lr, MLP_nhidden, activation, crash_duplicate_ratio)
# when retrain
save_path = './retrain_saved_model/nsamples{}_nparticle{}_tau{}_{}_lr{}_h{}_{}_crash_{}_checkpoint.pth'.format( nsamples, nparticle, tau_long, optimizer,
                                                 lr, MLP_nhidden, activation, crash_duplicate_ratio)
# when retrain
loss_curve = 'nsamples{}_nparticle{}_tau{}_{}_{}_lr{}_h{}_{}_crash_{}_loss.txt'.format(nsamples, nparticle,  tau_long, tau_short, optimizer, lr, MLP_nhidden, activation, crash_duplicate_ratio)

uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])
base_dir = uppath(__file__, 1)
init_path = base_dir + '/init_config/'
init_test_path = base_dir + '/init_config_for_testset/'
filename = 'tmp/nparticle{}_T{}_tau{}'.format(nparticle, temp[0], tau_long)

torch.autograd.set_detect_anomaly(True)

# # for train
# MD_learner = MD_learner(linear_integrator_obj, pair_wise_HNN_obj, phase_space, init_path)
# MD_learner.load_checkpoint(load_path)
# MD_learner.train_valid_epoch(save_path, best_model_path, loss_curve)

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

# for crash relearner
MD_relearner = MD_crash_relearner(linear_integrator_obj, pair_wise_HNN_obj, phase_space, init_path)
MD_relearner.load_checkpoint(load_path)
MD_relearner.train_valid_epoch(save_path, best_model_path, loss_curve)
