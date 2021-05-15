from data_io import data_io
from phase_space import phase_space
from check4particle_crash_dummy import check4particle_crash_dummy
import linear_integrator
from linear_velocity_verlet import linear_velocity_verlet
from fields_HNN import fields_HNN
from fields_MLP import fields_MLP as net
import torch.optim as optim
from loss  import qp_MSE_loss
import torch
import math

if __name__ == '__main__':

    seed = 9372211
    torch.manual_seed(seed)

    data_io = data_io()
    phase_space = phase_space()

    # init_q = torch.tensor([[[-1.9, -1.3], [1.9, -0.9]], [[-1.9, -1.3], [1.2, -0.9]]])
    # init_p = torch.tensor([[[0, 0], [0, 0]], [[0, 0], [0, 0]]])

    qp_list, tau_short, tau_long, boxsize = data_io.read_trajectory_qp('n2T0.1.pt')
    # qp_list shape = [nsamples, (q,p), trajectory (input,label), nparticle, DIM]

    qp_list_input = qp_list[:, :, 0, :, :]
    qp_list_label = qp_list[:, :, 1, :, :]

    phase_space.set_boxsize(boxsize)

    pthrsh = math.sqrt(2*1.0)*math.sqrt( -1. * math.log(math.sqrt(2*math.pi)*1e-6))
    ethrsh = 1e2

    crash_chker = check4particle_crash_dummy(linear_velocity_verlet, ethrsh, pthrsh)
    linear_integrator = linear_integrator.linear_integrator(linear_velocity_verlet,crash_chker)

    net1 = net(36, 128) # 36 = 18 grids(at t=0) + 18 grids(at t=tau)
    net2 = net(36, 128)
    hamiltonian_obj = fields_HNN(net1, net2, linear_integrator)

    opt = optim.SGD(hamiltonian_obj.net_parameters(),lr=0.01)
    hamiltonian_obj.set_tau_short(tau_short)

    hamiltonian_obj.train()
    train_loss = 0.
    train_qloss = 0.
    train_ploss = 0.

    for e in range(10):

        opt.zero_grad()
        # clear out the gradients of all variables in this optimizer (i.e. w,b)

        # input shape, [nsamples, (q,p)=2, nparticle, DIM]
        phase_space.set_q(qp_list_input[:, 0, :, :])
        phase_space.set_p(qp_list_input[:, 1, :, :])

        qp_list, crash_idx = linear_integrator.one_step(hamiltonian_obj, phase_space, tau_long)
        # qp_list shape, [nsamples, (q,p)=2, nparticle, DIM]

        q_predict = qp_list[:, 0, :, :]; p_predict = qp_list[:, 1, :, :]
        # q_predict shape, [nsamples, nparticle, DIM]

        q_label = qp_list_label[:, 0, :, :]; p_label = qp_list_label[:, 1, :, :]
        # q_label shape, [nsamples, nparticle, DIM]

        train_predict = (q_predict, p_predict)
        train_label = (q_label, p_label)

        loss, qloss, ploss = qp_MSE_loss(train_predict, train_label)
        loss.backward()
        opt.step()

        print('{} epoch:'.format(e + 1), 'train_loss:{:.6f}'.format(loss))