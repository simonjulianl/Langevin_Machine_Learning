import torch

class ML_parameters:

    ratio = 0.8
    lr = 0.01
    nepoch = 20
    optimizer = 'Adam'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # MLP network parameters
    MLP_input = 5
    MLP_nhidden = 32

    # CNN network parameters

    # optimizer parameters

