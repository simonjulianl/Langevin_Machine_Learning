import math
import torch
import phase_space
import linear_integrator
from check4particle_crash import check4particle_crash
from linear_velocity_verlet import linear_velocity_verlet
from data_io import data_io
from pairwise_HNN import pairwise_HNN
from pairwise_MLP import pairwise_MLP as net
#from pairwise_MLP_dummy import pairwise_MLP_dummy as net

if __name__ == '__main__':

    torch.manual_seed(699) # HK

    nparticle = 4
    rho = 0.1
    tau_cur = 0.1
    tau_long = 0.1
    boxsize = math.sqrt(nparticle / rho)
    net1 = net(5,64)
    net2 = net(5,64)
    hamiltonian = pairwise_HNN(net1,net2) # 5 : input ; 64 : channels
    hamiltonian.set_tau(tau_long)
    phase_space = phase_space.phase_space()

    init_qp, _, _, _ = data_io.read_trajectory_qp('n4T0.03seed6325nsamples10.pt')
    # init_qp.shape = [nsamples, (q, p), 1, nparticle, DIM]

    init_q = torch.squeeze(init_qp[:,0,:,:,:], dim=1)
    # init_q.shape = [nsamples, nparticle, DIM]

    init_p = torch.squeeze(init_qp[:,1,:,:,:], dim=1)

    phase_space.set_q(init_q)
    phase_space.set_p(init_p)
    phase_space.set_boxsize(boxsize)

    pthrsh = math.sqrt(2*1.0)*math.sqrt( -1. * math.log(math.sqrt(2*math.pi)*1e-6))
    ethrsh = 1e2 # HK
    #
    # for param in net1.parameters():
    #     param.requires_grad = False
    #
    # for param in net2.parameters():
    #     param.requires_grad = False

    crash_chker = check4particle_crash(linear_velocity_verlet, ethrsh, pthrsh)
    linear_integrator = linear_integrator.linear_integrator(linear_velocity_verlet,crash_chker)

    #hamiltonian.eval() # HK

    #with torch.no_grad():
    # 100 : every 100 trajectory saved,  1 :  the period of which qp_list append
    qp_list, crash_flag = linear_integrator.nsteps(hamiltonian, phase_space, tau_cur, 10000, 1)
