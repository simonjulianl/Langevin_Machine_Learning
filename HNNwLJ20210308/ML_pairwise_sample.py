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
tau_long = MD_parameters.tau_long
tau_short = MD_parameters.tau_short
lr = ML_parameters.lr
optimizer = ML_parameters.optimizer
MLP_nhidden = ML_parameters.MLP_nhidden
activation = ML_parameters.activation
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
best_model_path = './saved_model/nsamples{}_nparticle{}_tau{}_{}_lr{}_h{}_{}_checkpoint_best.pth'.format( nsamples, nparticle, tau_long, optimizer,
                                                     lr, MLP_nhidden, activation)

#change path when retrain
save_path = './saved_model/nsamples{}_nparticle{}_tau{}_{}_lr{}_h{}_{}_checkpoint.pth'.format( nsamples, nparticle, tau_long, optimizer,
                                                 lr, MLP_nhidden, activation)
#change path when retrain
loss_curve = 'nsamples{}_nparticle{}_tau{}_{}_{}_lr{}_h{}_{}_loss.txt'.format(nsamples, nparticle,  tau_long, tau_short, optimizer, lr, MLP_nhidden, activation)

uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])
base_dir = uppath(__file__, 1)
init_path = base_dir + '/init_config/'
init_test_path = base_dir + '/init_config_for_testset/'

torch.autograd.set_detect_anomaly(True)

# for train
# MD_learner = MD_learner(linear_integrator_obj, pair_wise_HNN_obj, phase_space, init_path)
# MD_learner.load_checkpoint(load_path)
# MD_learner.train_valid_epoch(save_path, best_model_path, loss_curve)

# for test
MD_tester = MD_tester(linear_integrator_obj, pair_wise_HNN_obj, phase_space, init_test_path, load_path)

init_q = MD_tester.test_data[:, 0]
init_p = MD_tester.test_data[:, 1]

init_q = torch.unsqueeze(init_q, dim=0)
init_p = torch.unsqueeze(init_p, dim=0)

q_pred, p_pred = MD_tester.step()

q_hist = torch.cat((init_q, q_pred.cpu()), dim=0)
p_hist = torch.cat((init_p, p_pred.cpu()), dim=0)

q_hist = torch.unsqueeze(q_hist, dim=1)
p_hist = torch.unsqueeze(p_hist, dim=1)

qp_hist = torch.cat((q_hist, p_hist), dim=1)

base_library = os.path.abspath('gold_standard')
torch.save(qp_hist, base_library + '/nparticle{}_T{}_ts{}_iter{}_vv_predicted.pt'.format(nparticle,temp[0],tau_long,ML_iterations))
