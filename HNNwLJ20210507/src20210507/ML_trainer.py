from parameters.MD_parameters         import MD_parameters
from parameters.ML_parameters         import ML_parameters
from phase_space                      import phase_space
from utils.make_hamiltonian           import make_hamiltonian
from integrator.linear_integrator     import linear_integrator
from HNN.data_loader                  import data_loader
from HNN.data_loader                  import my_data
from HNN.MD_learner                   import MD_learner
from utils.check4particle_crash_dummy import check4particle_crash_dummy as crsh_chker

import sys
import torch
import math

if __name__=='__main__':
    # need to load MD json file to get hamiltonian type
    # run something like this
    # python ML_trainer.py ../data/gen_by_ML/basename/MD_config.dict  ../data/gen_by_ML/basename/ML_config.dict

    argv = sys.argv
    MDjson_file = argv[1]
    MLjson_file = argv[2]

    MD_parameters.load_dict(MDjson_file)
    ML_parameters.load_dict(MLjson_file)

    seed = ML_parameters.seed
    torch.manual_seed(seed)  

    # io varaiables
    train_filename = ML_parameters.train_filename
    val_filename   = ML_parameters.valid_filename  # read the same data
    test_filename  = val_filename
    train_pts      = ML_parameters.train_pts
    val_pts        = ML_parameters.valid_pts
    test_pts       = val_pts

    # crash checker variables
    pthrsh = math.sqrt(2*1.0)*math.sqrt( -1. * math.log(math.sqrt(2*math.pi)*1e-6))
    ethrsh = 1e2

    # MD variables
    hamiltonian_type = MD_parameters.hamiltonian_type 
    tau_long = MD_parameters.tau_long
    tau_short = MD_parameters.tau_short

    phase_space = phase_space.phase_space()

    crash_checker = crsh_chker(MD_parameters.integrator_method_backward,ethrsh,pthrsh) 

    linear_integrator_obj = linear_integrator( MD_parameters.integrator_method, crash_checker) 

    hamiltonian_obj = make_hamiltonian(hamiltonian_type, linear_integrator_obj,
                                       tau_short, tau_long, ML_parameters)

    my_data_obj = my_data(train_filename, val_filename, test_filename, train_pts, val_pts, test_pts)
  
    loader  = data_loader(my_data_obj, ML_parameters.batch_size)
    
    # ML variables
    load_model_file = ML_parameters.ML_chk_pt_filename
    lr_decay_step = ML_parameters.lr_decay_step  
    lr_decay_rate = ML_parameters.lr_decay_rate 

    # create parameters from two models in one optimizer
    opt,sch = ML_parameters.opt.create(hamiltonian_obj.net_parameters(), lr_decay_step, lr_decay_rate)
    
    MD_learner = MD_learner(linear_integrator_obj, hamiltonian_obj, phase_space, opt, sch, loader, load_model_file)
    MD_learner.nepoch(ML_parameters.nepoch, ML_parameters.write_chk_pt_filename, ML_parameters.write_loss_filename ) 
    hamiltonian_obj.show_total_nn_time()