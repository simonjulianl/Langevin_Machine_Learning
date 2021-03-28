import torch
import sys
import os

sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("../parameters"))

from MC_parameters import MC_parameters
from MD_parameters import MD_parameters
from ML_parameters import ML_parameters

if __name__ == '__main__':

    nsamples = MD_parameters.nsamples
    nparticle = MC_parameters.nparticle
    tau_long = MD_parameters.tau_long
    lr = ML_parameters.lr
    optname = ML_parameters.opt.name()
    MLP_nhidden = ML_parameters.MLP_nhidden
    activation = ML_parameters.activation

    root_train_path = '../saved_model/'
    root_retrain_path = '../retrain_saved_model/'

    prefix = 'nsamples{}_nparticle{}_tau{}_{}_lr{}_h{}_{}'.format(nsamples, nparticle, tau_long, optname, lr,
                                                                  MLP_nhidden, activation)

    # best_model_path = root_train_path + prefix + '_checkpoint_best.pth'
    # load_path = root_train_path + prefix + '_checkpoint.pth'
    best_model_path = root_retrain_path + prefix + '_crash_0.4_checkpoint_best.pth'
    load_path = root_retrain_path + prefix + '_crash_0.4_checkpoint.pth'

    print("=> loading checkpoint '{}'".format(load_path))
    checkpoint = torch.load(load_path)[0]
    print(checkpoint)
    epoch = checkpoint['epoch']
    print('Previously trained epoch state_dict loaded...')
    print('epoch', epoch)
    best_validation_loss = checkpoint['best_validation_loss']
    print('best_validation_loss', best_validation_loss)
