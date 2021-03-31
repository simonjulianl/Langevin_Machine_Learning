from hamiltonian.hamiltonian         import hamiltonian
from hamiltonian.kinetic_energy      import kinetic_energy
from hamiltonian.lennard_jones       import lennard_jones 
from parameters.MD_parameters        import MD_parameters
from phase_space                     import phase_space
from integrator.linear_integrator    import linear_integrator
from utils.data_io                   import data_io

import psutil
import shutil
import torch

if __name__=='__main__':

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

    data_io_obj = data_io()
     
    phase_space = phase_space.phase_space()
    linear_integrator_obj = linear_integrator( MD_parameters.integrator_method )
    
    noMLhamiltonian = hamiltonian()
    ke = kinetic_energy()
    pe = lennard_jones()
    noMLhamiltonian.append(ke)
    noMLhamiltonian.append(pe)
    
    init_qp = data_io_obj.read_init_qp(init_filename)
   
    phase_space.set_q(init_qp[0])
    phase_space.set_p(init_qp[1])
    
    # write file
    for i in range(n_out_files):
        print('save file ',i)
        qp_list = linear_integrator_obj.nsteps(noMLhamiltonian, phase_space, tau_cur, save2file_strike, append_strike)
        qp_list = torch.stack(qp_list)
        print('i', i, 'memory % used:', psutil.virtual_memory()[2])
        tmp_filename = data_filenames + '_id' + str(i) + '.pt'
        data_io_obj.write_trajectory_qp(tmp_filename, qp_list)
    
    # cp init file in same training filename folder
    shutil.copy2(init_filename, MD_parameters.data_path)