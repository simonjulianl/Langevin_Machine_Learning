import sys
import os
sys.path.append(os.path.abspath("./parameters"))

from HNN.field_HNN import field_HNN
from HNN.models import fields_unet
# from HNN.models import pair_wise_zero
from HNN.MD_learner import MD_learner
from parameters.MC_parameters import MC_parameters
from parameters.MD_parameters import MD_parameters
from parameters.ML_parameters import ML_parameters
from phase_space import phase_space
from integrator import linear_integrator
import torch

nsamples = MD_parameters.nsamples
nparticle = MC_parameters.nparticle
npixels = MD_parameters.npixels
tau_long = MD_parameters.tau_long
lr = ML_parameters.lr
optimizer = ML_parameters.optimizer
cnn_input = ML_parameters.cnn_input
cnn_nhidden = ML_parameters.cnn_nhidden
activation = ML_parameters.activation
tau_short = MD_parameters.tau_short

seed = 9372211
torch.manual_seed(seed)
# torch.backends.cudnn.deterministic = True  # Processing speed may be lower then when the models functions nondeterministically.
# torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed_all(seed)

phase_space = phase_space.phase_space()
linear_integrator_obj = linear_integrator( MD_parameters.integrator_method )
field_HNN_obj = field_HNN(fields_unet(), linear_integrator_obj) # in_channels, n_channels, out_channels

# check gpu available
print('__Number CUDA Devices:', torch.cuda.device_count())
print('__Devices')
print('Active CUDA Device: GPU', torch.cuda.current_device())
print ('Available devices ', torch.cuda.device_count())
print ('Current cuda device ', torch.cuda.current_device())
# print('GPU available', torch.cuda.get_device_name(device))

load_path = './saved_model/nsamples{}_nparticle{}_tau{}_{}_lr{}_h{}_{}_checkpoint.pth'.format( nsamples, nparticle, tau_long, optimizer,
                                                 lr, cnn_nhidden, activation)
best_model_path = './saved_model/nsamples{}_nparticle{}_tau{}_{}_lr{}_h{}_{}_checkpoint_best.pth'.format( nsamples, nparticle, tau_long, optimizer,
                                                     lr, cnn_nhidden, activation)

#change path when retrain빠
save_path = './saved_model/nsamples{}_nparticle{}_tau{}_{}_lr{}_h{}_{}_checkpoint.pth'.format( nsamples, nparticle, tau_long, optimizer,
                                                 lr, cnn_nhidden, activation)
#change path when retrain
loss_curve = 'nsamples{}_nparticle{}_tau{}_{}_{}_lr{}_h{}_{}_loss.txt'.format(nsamples, nparticle,  tau_long, tau_short, optimizer, lr, cnn_nhidden, activation)

uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])
base_dir = uppath(__file__, 1)
init_path = base_dir + '/init_config/'

torch.autograd.set_detect_anomaly(True)

MD_learner = MD_learner(linear_integrator_obj, field_HNN_obj, phase_space, init_path)
MD_learner.load_checkpoint(load_path)

MD_learner.train_valid_epoch(save_path, best_model_path, loss_curve)
# pred = MD_learner.pred_qnp(filename ='./init_config/N_particle{}_samples{}_rho0.1_T0.04_pos_sampled.pt'.format(nparticle, nsamples_label))