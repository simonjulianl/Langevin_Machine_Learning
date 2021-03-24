import torch
import torch.optim as optim

class optimizer:

    def __init__(self, lr):

        self.lr = lr
        self.opttype = optim.Adam
        self.optname = self.opttype.__name__

    def create(self, para):
        return self.opttype(para, self.lr)

    def name(self):
        return self.optname

class ML_parameters:

    seed = 9372211
    lr = 0.0001
    nepoch = 5
    opt = optimizer(lr)
    activation = 'tanh'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # MLP network parameters
    MLP_input = 5
    MLP_nhidden = 128

    # CNN network parameters
    cnn_input = 2
    cnn_nhidden = 32

    # optimizer parameters

