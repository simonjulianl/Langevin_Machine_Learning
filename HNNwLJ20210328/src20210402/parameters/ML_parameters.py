import torch
import torch.optim as optim
from HNN.optimizer import optimizer

class ML_parameters:

    check_path = '../data/trainning_checkpoint/'
    check_file = None # if None then do not load model
    #check_file = '../data/.. . ..pth'

    train_data_file = '../data/training_data/n2/nparticle2_rho0.1_allT_train_sampled.pt'
    valid_data_file = '../data/training_data/n2/nparticle2_rho0.1_allT_valid_sampled.pt'

    seed = 9372211

    # optimizer parameters
    lr = 0.0001
    nepoch = 5
    opt_op = optim.Adam
    opt = optimizer(opt_op,lr)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # MLP network parameters
    MLP_input = 5
    MLP_nhidden = 128
    activation = 'tanh'

    # CNN network parameters
    cnn_input = 2
    cnn_nhidden = 32


