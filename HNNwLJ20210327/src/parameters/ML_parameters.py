import torch
import torch.optim as optim
from HNN.optimizer import optimizer

class ML_parameters:

    lr = 0.0001
    nepoch = 5
    opt_op = optim.Adam
    opt = optimizer(opt_op,lr)
    activation = 'tanh'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # MLP network parameters
    MLP_input = 5
    MLP_nhidden = 128

    # CNN network parameters
    cnn_input = 2
    cnn_nhidden = 32

    # optimizer parameters

