from parameters.MD_parameters        import MD_parameters
from parameters.ML_parameters        import ML_parameters
from phase_space                     import phase_space
from integrator.linear_integrator    import linear_integrator
from HNN.data_loader                 import data_loader
from HNN.data_loader                 import my_data
from HNN.MD_learner                  import MD_learner
from HNN.checkpoint                import checkpoint
from HNN.models.pairwise_MLP      import pairwise_MLP

import torch

if __name__=='__main__':

    tau_long         = MD_parameters.tau_long
    tau_cur          = tau_long

    # io varaiables
    train_filename = ML_parameters.train_filename
    val_filename   = train_filename # read the same data
    test_filename  = train_filename
    train_pts      = ML_parameters.train_pts
    val_pts        = train_pts
    test_pts       = train_pts

    dataset = my_data(train_filename,val_filename,test_filename,train_pts,val_pts,test_pts)
    loader  = data_loader(dataset, ML_parameters.batch_size)

    phase_space = phase_space.phase_space()
    hamiltonian_obj = MD_parameters.hamiltonian_obj
    linear_integrator_obj = linear_integrator( MD_parameters.integrator_method, MD_parameters.integrator_method_backward )
 
    seed = ML_parameters.seed
    torch.manual_seed(seed) # cpu

    net = pairwise_MLP()
    opt = ML_parameters.opt.create(hamiltonian_obj.net_parameters())
    chk_pt = checkpoint(ML_parameters.check_path, net, opt)

    if ML_parameters.check_file is not None:
        chk_pt.load_checkpoint(ML_parameters.check_file)

    MD_learner = MD_learner(linear_integrator_obj, hamiltonian_obj, phase_space, opt, loader, chk_pt)
    MD_learner.nepoch(ML_parameters.save_filename, ML_parameters.write_loss_filename )
