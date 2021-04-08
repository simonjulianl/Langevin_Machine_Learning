
class ML_config:

    # chk_pt_file = None # if None then do not load model
    # load_check_file = '../data/.. . ..pth'

    train_filename          = '../data/training_data/combined/n2rho0.1ts0.001tl0.1_train_sampled.pt'
    write_chk_pt_filename   = '../data/training_checkpoint/n2ts0.001tl0.1.pth'        # filename to save checkpoints
    write_loss_filename     = '../data/training_loss/n2ts0.001tl0.1.txt'              # filename to save loss values

    train_pts      = 2   # selected number of nsamples
    seed = 9372211

    # optimizer parameters
    lr = 0.0001
    nepoch = 3
    batch_size = 2

    # MLP network parameters
    MLP_input = 5
    MLP_nhidden = 128
    activation = 'tanh'

