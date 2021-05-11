import json
import torch.optim as optim
from HNN.optimizer import optimizer

class ML_parameters:

    ML_chk_pt_filename      = None            # if None then do not load model

    train_filename          = None
    valid_filename          = None
    write_chk_pt_filename   = None            # filename to save checkpoints
    write_loss_filename     = None            # filename to save loss values

    train_pts               = None            # selected number of nsamples
    valid_pts               = None            # selected number of nsamples
    seed                    = None

    # optimizer parameters
    lr                      = None
    lr_decay_step           = None
    lr_decay_rate           = None
    nepoch                  = None
    batch_size              = None

    # MLP network parameters
    MLP_input               = None
    MLP_nhidden             = None
    activation              = None

    # CNN network parameters
    gridL                   = None
    cnn_channels            = None

    #opt_op                  = optim.Adam      # optimizer option
    opt_op                  = optim.SGD        # optimizer option
    opt                     = None             # construct optimizer
    # opt = optimizer(opt_op,lr)

    @staticmethod
    def load_dict(json_filename):
        with open(json_filename) as f:
            data = json.load(f)

        ML_parameters.ML_chk_pt_filename        = data['ML_chk_pt_filename']
        ML_parameters.train_filename            = data['train_filename']
        ML_parameters.valid_filename            = data['valid_filename']
        ML_parameters.write_chk_pt_filename     = data['write_chk_pt_filename']
        ML_parameters.write_loss_filename       = data['write_loss_filename']

        ML_parameters.train_pts                 = data['train_pts']
        ML_parameters.valid_pts                 = data['valid_pts']
        ML_parameters.seed                      = data['seed']
        ML_parameters.lr                        = data['lr']
        ML_parameters.lr_decay_step             = data['lr_decay_step']
        ML_parameters.lr_decay_rate             = data['lr_decay_rate']
        ML_parameters.nepoch                    = data['nepoch']
        ML_parameters.batch_size                = data['batch_size']
        ML_parameters.MLP_input                 = data['MLP_input']
        ML_parameters.MLP_nhidden               = data['MLP_nhidden']
        ML_parameters.activation                = data['activation']
        ML_parameters.gridL                     = data['gridL']
        ML_parameters.cnn_channels               = data['cnn_channels']

        ML_parameters.opt = optimizer(ML_parameters.opt_op, ML_parameters.lr)
