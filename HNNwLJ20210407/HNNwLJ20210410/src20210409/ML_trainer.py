from parameters.MD_parameters        import MD_parameters
from parameters.ML_parameters        import ML_parameters
from phase_space                     import phase_space
from utils.make_hamiltonian          import make_hamiltonian
from integrator.linear_integrator    import linear_integrator
from HNN.data_loader                 import data_loader
from HNN.data_loader                 import my_data
from HNN.MD_learner                  import MD_learner


import sys
import torch

if __name__=='__main__':
    # need to load MD json file also to get tau_cur in pairwise_HNN
    # run something like this
    # python ML_trainer.py ../data/training_data/n2run@@/MD_config.dict  ../data/training_checkpoint/n2run@@/ML_config.dict

    argv = sys.argv
    MDjson_file = argv[1]
    MLjson_file = argv[2]

    MD_parameters.load_dict(MDjson_file)
    ML_parameters.load_dict(MLjson_file)

    # io varaiables
    train_filename = ML_parameters.train_filename
    val_filename   = train_filename # read the same data
    test_filename  = train_filename
    train_pts      = ML_parameters.train_pts
    val_pts        = train_pts
    test_pts       = train_pts

    phase_space = phase_space.phase_space()
    hamiltonian_obj = make_hamiltonian(MD_parameters.hamiltonian_type, MD_parameters.tau_long, ML_parameters)
    linear_integrator_obj = linear_integrator( MD_parameters.integrator_method, MD_parameters.integrator_method_backward )

    seed = ML_parameters.seed
    torch.manual_seed(seed) # cpu

    my_data_obj = my_data(train_filename, val_filename, test_filename, train_pts, val_pts, test_pts)
    loader  = data_loader(my_data_obj, ML_parameters.batch_size)

    # create parameters from two models in one optimizer
    opt = ML_parameters.opt.create(hamiltonian_obj.net_parameters())

    MD_learner = MD_learner(linear_integrator_obj, hamiltonian_obj, phase_space, opt, loader )
    MD_learner.nepoch(ML_parameters.nepoch, ML_parameters.write_chk_pt_filename, ML_parameters.write_loss_filename )
