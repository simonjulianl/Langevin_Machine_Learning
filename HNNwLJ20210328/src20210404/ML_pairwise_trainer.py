from parameters.MD_parameters      import MD_parameters
from HNN.MD_learner                import MD_learner
from HNN.data_handler              import data_handler
from phase_space.phase_space       import phase_space
from integrator.linear_integrator  import linear_integrator

from parameters.ML_parameters      import ML_parameters
from HNN.checkpoint                import checkpoint
from HNN.pair_wise_HNN             import pair_wise_HNN
from HNN.models.pair_wise_MLP      import pair_wise_MLP
# from HNN.models.pair_wise_zero    import pair_wise_zero

import torch

seed = ML_parameters.seed
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

dataset = data_handler()
phase_space = phase_space()
linear_integrator_obj = linear_integrator( MD_parameters.integrator_method, MD_parameters.integrator_method_backward )

net = pair_wise_MLP()
opt = ML_parameters.opt.create(net.parameters())
chk_pt = checkpoint(ML_parameters.check_path, net, opt)

pair_wise_HNN_obj = MD_parameters.hamiltonian_obj
# pair_wise_HNN_obj = pair_wise_HNN(net.to(ML_parameters.device))

if ML_parameters.check_file is not None:
    chk_pt.load_checkpoint(ML_parameters.check_file)

# training data
dataset.load(ML_parameters.train_data_file)
qp_input_train, qp_label_train = dataset._shuffle(dataset.qp_list_input, dataset.qp_list_label)
print('input qp shape', qp_input_train.shape)
print('label qp shape', qp_label_train.shape)

# validation data
dataset.load(ML_parameters.valid_data_file)
qp_input_valid, qp_label_valid = dataset._shuffle(dataset.qp_list_input, dataset.qp_list_label)

# combine train/valid data
dataset = (torch.stack((qp_input_train, qp_label_train)), torch.stack((qp_input_valid, qp_label_valid)))

# use anomaly detection for debugging
torch.autograd.set_detect_anomaly(True)

MD_learner = MD_learner(linear_integrator_obj, pair_wise_HNN_obj, phase_space, opt, dataset, chk_pt)
MD_learner.nepoch(1, ML_parameters.save_filename, ML_parameters.write_loss_filename)
