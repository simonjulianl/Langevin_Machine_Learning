import json
import torch.optim as optim
from HNN.optimizer import optimizer

class ML_parameters:

    chk_pt_file             = None            # if None then do not load model

    train_filename          = None
    write_chk_pt_filename   = None            # filename to save checkpoints
    write_loss_filename     = None            # filename to save loss values

    train_pts               = None            # selected number of nsamples
    seed                    = None

    # optimizer parameters
    lr                      = None
    nepoch                  = None
    batch_size              = None

    # MLP network parameters
    MLP_input               = None
    MLP_nhidden             = None
    activation              = None

    # CNN network parameters
    cnn_input               = None
    cnn_nhidden             = None

    opt_op                  = optim.SGD      # optimizer option
    opt                     = None            # construct optimizer
    # opt = optimizer(opt_op,lr)

    @staticmethod
    def load_dict(json_filename):
        with open(json_filename) as f:
            data = json.load(f)

        ML_parameters.chk_pt_file               = data['chk_pt_file']
        ML_parameters.train_filename            = data['train_filename']
        ML_parameters.write_chk_pt_filename     = data['write_chk_pt_filename']
        ML_parameters.write_loss_filename       = data['write_loss_filename']

        ML_parameters.train_pts                 = data['train_pts']
        ML_parameters.seed                      = data['seed']
        ML_parameters.lr                        = data['lr']
        ML_parameters.nepoch                    = data['nepoch']
        ML_parameters.batch_size                = data['batch_size']
        ML_parameters.MLP_input                 = data['MLP_input']
        ML_parameters.MLP_nhidden               = data['MLP_nhidden']
        ML_parameters.activation                = data['activation']

        ML_parameters.opt = optimizer(ML_parameters.opt_op, ML_parameters.lr)