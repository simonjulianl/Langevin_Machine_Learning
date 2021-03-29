import sys
import os

sys.path.append(os.path.abspath("./parameters"))

from hamiltonian.hamiltonian    import hamiltonian
from hamiltonian.kinetic_energy import kinetic_energy
from hamiltonian.lennard_jones  import lennard_jones 
from parameters.MC_parameters   import MC_parameters
from parameters.MD_parameters   import MD_parameters
from phase_space                import phase_space
from integrator                 import linear_integrator
from utils.data_io              import data_io

import torch
import time

mass = MC_parameters.mass
nsamples = MD_parameters.nsamples
nparticle = MC_parameters.nparticle
rho = MC_parameters.rho
temp = MD_parameters.temp_list
mode = MC_parameters.mode

tau_short = MD_parameters.tau_short
append_strike = MD_parameters.append_strike
niter_tau_long = MD_parameters.niter_tau_long
save2file_strike = MD_parameters.save2file_strike
tau_long = MD_parameters.tau_long
niter_tau_short = MD_parameters.niter_tau_short

tau_cur = tau_short
nfile = niter_tau_short // save2file_strike

print('tau long, tau short, niter tau long, niter tau short, max. times, nfile')
print(tau_long, tau_short, niter_tau_long, niter_tau_short, tau_long * niter_tau_long, nfile)

uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])
base_dir = uppath(__file__, 2)
root_path =  base_dir + '/data/'
write_path =  base_dir + '/data/training_data/'
tmp_path =  base_dir + '/data/tmp/'

tmp = 'tmp/nparticle{}_T{}_tau{}'.format(nparticle, temp[0], tau_cur)
init_filename = 'init_config/nparticle{}_new_nsim_rho{}_T{}_pos_{}_sampled.pt'.format(nparticle, rho, temp[0], mode)
write_filename = 'training_data/nparticle{}_T{}_tl{}_ts{}_iter{}_vv_{}sampled.pt'.format(nparticle,temp[0],tau_long,tau_short,niter_tau_short, nsamples)

if not os.path.exists(write_path):
                os.makedirs(write_path)

if not os.path.exists(tmp_path):
    os.makedirs(tmp_path)

data_io_obj = data_io(root_path)
phase_space = phase_space.phase_space()
linear_integrator_obj = linear_integrator( MD_parameters.integrator_method )

noMLhamiltonian = hamiltonian()
ke = kinetic_energy(mass)
pe = lennard_jones()
noMLhamiltonian.append(ke)
noMLhamiltonian.append(pe)

init_qp = data_io_obj.read_init_qp(init_filename)

phase_space.set_q(init_qp[0])
phase_space.set_p(init_qp[1])

# e = noMLhamiltonian.total_energy(phase_space)
# print('energy ', e)

# write file
for i in range(nfile):
    print('save file ',i)
    qp_list = linear_integrator_obj.nsteps(noMLhamiltonian, phase_space, tau_cur, save2file_strike, append_strike)
    qp_list = torch.stack(qp_list)
    # print('write qp', qp_list.shape)
    tmp_filename = tmp + str(i) + '.pt'
    data_io_obj.write_trajectory_qp(tmp_filename, qp_list)


# concat file
qp_list = []
for i in range(nfile):

    tmp_filename = tmp + str(i) + '.pt'
    qp = data_io_obj.read_trajectory_qp(tmp_filename)

    assert (torch.all(torch.eq(torch.load(root_path + tmp_filename), qp))), 'write and read file not match'
    # print('read qp', qp)
    qp_list.append(qp)

tensor_qp_list = torch.cat(qp_list)

init_qp = torch.unsqueeze(init_qp, dim=0)

qp_hist = torch.cat((init_qp, tensor_qp_list), dim=0)

data_io_obj.write_trajectory_qp(write_filename, qp_hist)

# remove files
for z in range(nfile):
    os.remove( root_path + tmp + str(z) + '.pt' )
