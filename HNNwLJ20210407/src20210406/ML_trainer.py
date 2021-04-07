from parameters.MD_parameters        import MD_parameters
from parameters.ML_parameters        import ML_parameters
from phase_space                     import phase_space
from integrator.linear_integrator    import linear_integrator
from HNN.data_loader                 import data_loader
from HNN.data_loader                 import my_data
from HNN.MD_learner                  import MD_learner
from HNN.checkpoint                  import checkpoint
from HNN.models.pairwise_MLP         import pairwise_MLP

import torch

if __name__=='__main__':

    # io varaiables
    train_filename = ML_parameters.train_filename
    val_filename   = train_filename # read the same data
    test_filename  = train_filename
    train_pts      = ML_parameters.train_pts
    val_pts        = train_pts
    test_pts       = train_pts

    phase_space = phase_space.phase_space()
    hamiltonian_obj = MD_parameters.hamiltonian_obj
    linear_integrator_obj = linear_integrator( MD_parameters.integrator_method, MD_parameters.integrator_method_backward )

    my_data_obj = my_data(train_filename, val_filename, test_filename, train_pts, val_pts, test_pts)
    loader  = data_loader(my_data_obj, ML_parameters.batch_size)

    seed = ML_parameters.seed
    torch.manual_seed(seed) # cpu

    netA = pairwise_MLP()
    netB = pairwise_MLP()
    optA = ML_parameters.opt.create(hamiltonian_obj.netA.parameters())
    optB = ML_parameters.opt.create(hamiltonian_obj.netB.parameters())
    chk_pt = checkpoint(ML_parameters.chk_pt_path, netA, netB, optA, optB)

    # optimize parameters from two models in one optimizer
    opt = ML_parameters.opt.create(hamiltonian_obj.net_parameters())

    if ML_parameters.chk_pt_file is not None:
        chk_pt.load_checkpoint(ML_parameters.chk_pt_file)

    MD_learner = MD_learner(linear_integrator_obj, hamiltonian_obj, phase_space, opt, loader, chk_pt)
    MD_learner.nepoch(ML_parameters.write_chk_pt_filename, ML_parameters.write_loss_filename )
