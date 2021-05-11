from parameters.MD_parameters        import MD_parameters
from parameters.ML_parameters        import ML_parameters
from phase_space                     import phase_space
from integrator.linear_integrator    import linear_integrator
from utils.data_io                   import data_io
from utils.make_hamiltonian          import make_hamiltonian
from HNN.checkpoint                  import checkpoint
from utils.check4particle_crash      import check4particle_crash as crsh_chker 
# do we need to use the actual crash checker? this code is also use for long trajectory

import sys
import math
import psutil
import shutil
import torch

if __name__=='__main__':
    # need to load ML json file to get ML paramters
    # run something like this
    # python MD_sampler.py ../data/gen_by_MD/basename/MD_config.dict ../data/gen_by_MD/basename/ML_config.dict

    argv = sys.argv
    MDjson_file = argv[1]
    MLjson_file = argv[2]

    MD_parameters.load_dict(MDjson_file)
    ML_parameters.load_dict(MLjson_file)

    seed = ML_parameters.seed
    torch.manual_seed(seed)

    tau_short        = MD_parameters.tau_short
    append_strike    = MD_parameters.append_strike
    niter_tau_long   = MD_parameters.niter_tau_long
    save2file_strike = MD_parameters.save2file_strike
    tau_long         = MD_parameters.tau_long
    niter_tau_short  = MD_parameters.niter_tau_short

    # io varaiables
    MC_init_config_filename       = MD_parameters.MC_init_config_filename
    MD_output_basenames           = MD_parameters.MD_output_basenames
    print('MC output filename', MC_init_config_filename)

    # MD variables
    tau_cur          = tau_short
    tau_short        = tau_short
    tau_long         = tau_long
    hamiltonian_type = MD_parameters.hamiltonian_type 
    n_out_files      = niter_tau_short // save2file_strike

    # ML variables
    load_model_file  = ML_parameters.ML_chk_pt_filename

    # crash checker variables
    # T = 1.0 is given
    pthrsh = math.sqrt(2*1.0)*math.sqrt( -1. * math.log(math.sqrt(2*math.pi)*1e-6))
    ethrsh = 1e2

    phase_space = phase_space.phase_space()

    crash_checker = crsh_chker(MD_parameters.integrator_method_backward,ethrsh,pthrsh)

    linear_integrator_obj = linear_integrator( MD_parameters.integrator_method, crash_checker )

    hamiltonian_obj = make_hamiltonian(hamiltonian_type, linear_integrator_obj, tau_short, tau_long, ML_parameters)

    if hamiltonian_type != "noML":
        chk_pt = checkpoint(hamiltonian_obj.get_netlist())
        if load_model_file is not None: chk_pt.load_checkpoint(load_model_file)
        hamiltonian_obj.requires_grad_false()

    init_qp, _, _, boxsize = data_io.read_trajectory_qp(MC_init_config_filename)
    # init_qp.shape = [nsamples, (q, p), 1, nparticle, DIM]

    init_q = torch.squeeze(init_qp[:,0,:,:,:], dim=1)
    # init_q.shape = [nsamples, nparticle, DIM]

    init_p = torch.squeeze(init_qp[:,1,:,:,:], dim=1)
    # init_p.shape = [nsamples, nparticle, DIM]

    phase_space.set_q(init_q)
    phase_space.set_p(init_p)
    phase_space.set_boxsize(boxsize)

    crash_history = False

    # write file
    for i in range(n_out_files):

        qp_list, crash_flag = linear_integrator_obj.nsteps(hamiltonian_obj, phase_space, tau_cur, save2file_strike, append_strike)
        # qp_list is [nsamples, (q, p), nparticle, DIM] append to trajectory length number of items
        # crash_flag is true, some of nsamples have crashed qp_list
        # crash_flag is false, nsamples don't have crashed qp_list

        crash_history = crash_flag

        # if crash_flag is false, write file to save qp list in intermediate steps of integration
        if crash_history is False:
            print('no crash so that save file ', i)
            qp_list = torch.stack(qp_list, dim=2)
            # qp_list.shape = [nsamples, (q, p), trajectory length, nparticle, DIM]

            print('i', i, 'memory % used:', psutil.virtual_memory()[2], '\n')
            tmp_filename = MD_output_basenames + '_id' + str(i) + '.pt'
            data_io.write_trajectory_qp(tmp_filename, qp_list, boxsize, tau_short, tau_long)

    # cp init file in same training filename folder
    shutil.copy2(MC_init_config_filename, MD_parameters.MD_data_dir)
    print('file write dir:', MD_parameters.MD_data_dir)
