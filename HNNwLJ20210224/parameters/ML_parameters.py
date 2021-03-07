import torch

class ML_parameters:

    lr = 0.0001
    nepoch = 3
    optimizer = 'Adam'
    activation = 'tanh'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # MLP network parameters
    MLP_input = 5
    MLP_nhidden = 128

    # CNN network parameters

    # optimizer parameters

