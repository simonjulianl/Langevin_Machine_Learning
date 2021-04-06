from parameters.MD_parameters        import MD_parameters
from phase_space                     import phase_space
from integrator.linear_integrator    import linear_integrator
from utils.data_io                   import data_io

import sys
import psutil
import shutil
import torch

if __name__=='__main__':
    # run something like this
    # python MD_sampler.py ../data/training_data/n2run@@/MD_config.dict

    argv = sys.argv
    MDjson_file = argv[1]
    MD_parameters.load_dict(MDjson_file)

    tau_short        = MD_parameters.tau_short
    append_strike    = MD_parameters.append_strike
    niter_tau_long   = MD_parameters.niter_tau_long
    save2file_strike = MD_parameters.save2file_strike
    tau_long         = MD_parameters.tau_long
    niter_tau_short  = MD_parameters.niter_tau_short

    tau_cur          = tau_short
    n_out_files      = niter_tau_short // save2file_strike

    # io varaiables
    init_filename       = MD_parameters.init_qp_filename
    data_filenames      = MD_parameters.data_filenames

    phase_space = phase_space.phase_space()
    hamiltonian_obj = MD_parameters.hamiltonian_obj
    linear_integrator_obj = linear_integrator( MD_parameters.integrator_method, MD_parameters.integrator_method_backward )

    init_qp, _, _ = data_io.read_trajectory_qp(init_filename)
    # init_qp.shape = [nsamples, (q, p), 1, nparticle, DIM]

    init_q = torch.squeeze(init_qp[:,0,:,:,:], dim=1)
    # init_q.shape = [nsamples, nparticle, DIM]

    init_p = torch.squeeze(init_qp[:,1,:,:,:], dim=1)
    # init_p.shape = [nsamples, nparticle, DIM]

    phase_space.set_q(init_q)
    phase_space.set_p(init_p)
    
    # write file
    for i in range(n_out_files):
        print('save file ',i)
        qp_list = linear_integrator_obj.nsteps(hamiltonian_obj, phase_space, tau_cur, save2file_strike, append_strike)
        qp_list = torch.stack(qp_list, dim=2)
        # qp_list.shape = [nsamples, (q, p), trajectory length, nparticle, DIM]

        print('i', i, 'memory % used:', psutil.virtual_memory()[2])
        tmp_filename = data_filenames + '_id' + str(i) + '.pt'
        data_io.write_trajectory_qp(tmp_filename, qp_list, tau_short, tau_long)

    # cp init file in same training filename folder
    shutil.copy2(init_filename, MD_parameters.data_path)
    print('file write dir:', MD_parameters.data_path)