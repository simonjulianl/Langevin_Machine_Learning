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

import time

tau_short = MD_parameters.tau_short
tau_ratio = MD_parameters.tau_ratio
tau_long = MD_parameters.tau_long
tau_cur = tau_short # find gold standard
niter_tau_long = MD_parameters.niter_tau_long 
save2file_strike = MD_parameters.save2file_strike
niter_tau_short = MD_parameters.niter_tau_short
nfile = niter_tau_short // save2file_strike

uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])
base_dir = uppath(__file__, 2)
init_path =  base_dir + '../data/init_config/'
write_path =  base_dir + '../data/gold_standard/'
tmp_path =  base_dir + '../data/tmp/'

if not os.path.exists(write_path):
                os.makedirs(write_path)

if not os.path.exists(tmp_path):
    os.makedirs(tmp_path)

data_io_obj = data_io(init_path)
phase_space = phase_space.phase_space()
linear_integrator_obj = linear_integrator( MD_parameters.integrator_method )

noMLhamiltonian = hamiltonian()
ke = kinetic_energy(1.0)
pe = lennard_jones()
noMLhamiltonian.append(ke)
noMLhamiltonian.append(pe)

init_q, init_p = data_io_obj.read_init_qp('test.pt')

phase_space.set_q(init_q)
phase_space.set_p(init_p)

for i in range(nfile):
    qp_list = linear_integrator_obj.nsteps(noMLhamiltonian,phase_space,tau_cur,save2file_strike,tau_ratio)
    filename = 'test' + str(i) + '.pt'
    io.write_init_trajectory_qp(filename,qp_list)

