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

if not os.path.exists('./tmp/'):
    os.makedirs('./tmp/')

root_train_path = 'saved_model/'
root_retrain_path = './retrain_saved_model/'

prefix = 'nsamples{}_nparticle{}_tau{}_{}_lr{}_h{}_{}'.format( nsamples, nparticle, tau_long, optimizer, lr, cnn_nhidden, activation)
load_path = root_train_path + prefix + '_checkpoint.pth'

# path for train
best_model_path = root_train_path + prefix + '_checkpoint_best.pth'
save_path = root_train_path + prefix + '_checkpoint.pth'
loss_curve = prefix + '_loss.txt'


uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])
base_dir = uppath(__file__, 1)
init_path = base_dir + '/init_config/'

torch.autograd.set_detect_anomaly(True)

MD_learner = MD_learner(linear_integrator_obj, field_HNN_obj, phase_space, init_path)
MD_learner.load_checkpoint(load_path)

MD_learner.train_valid_epoch(save_path, best_model_path, loss_curve)
# pred = MD_learner.pred_qnp(filename ='./init_config/N_particle{}_samples{}_rho0.1_T0.04_pos_sampled.pt'.format(nparticle, nsamples_label))
