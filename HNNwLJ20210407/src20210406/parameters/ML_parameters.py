import torch
import torch.optim as optim
from HNN.optimizer import optimizer

class ML_parameters:

    check_path = '../data/training_checkpoint/'
    check_file = None # if None then do not load model
    # check_file = '../data/.. . ..pth'

    train_filename      = '../data/training_data/combined/n2rho0.1ts0.001tl0.1_train_sampled.pt'
    save_filename       = 'nparticle2_tau_lt0.1_st0.001.pth'
    write_loss_filename = '../data/training_loss/n2st0.001tl0.1.txt'

    train_pts      = 6   # selected number of nsamples
    seed = 9372211

    # optimizer parameters
    lr = 0.0001
    nepoch = 5
    opt_op = optim.Adam
    opt = optimizer(opt_op,lr)
    batch_size = 1

    # MLP network parameters
    MLP_input = 5
    MLP_nhidden = 128
    activation = 'tanh'

    # CNN network parameters
    cnn_input = 2
    cnn_nhidden = 32


