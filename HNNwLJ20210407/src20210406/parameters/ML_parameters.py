import torch
import torch.optim as optim
from HNN.optimizer import optimizer

class ML_parameters:

    chk_pt_path = '../data/training_checkpoint/'
    chk_pt_file = None # if None then do not load model
    # load_check_file = '../data/.. . ..pth'

    train_filename          = '../data/training_data/combined/n2rho0.1ts0.001tl0.1_train_sampled.pt'
    write_chk_pt_filename   = 'n2_tau_lt0.1_st0.001.pth'            # filename to save checkpoints
    write_loss_filename     = '../data/training_loss/n2st0.001tl0.1.txt'    # filename to save loss values

    train_pts      = 2   # selected number of nsamples
    seed = 9372211

    # optimizer parameters
    lr = 0.0001
    nepoch = 2
    opt_op = optim.Adam
    opt = optimizer(opt_op,lr)
    batch_size = 2

    # MLP network parameters
    MLP_input = 5
    MLP_nhidden = 128
    activation = 'tanh'

    # CNN network parameters
    cnn_input = 2
    cnn_nhidden = 32


