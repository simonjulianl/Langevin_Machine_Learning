import torch
import math
import sys
import matplotlib.pyplot as plt

if __name__ == '__main__':

    argv = sys.argv

    if len(argv) != 4:
        print('usage <programe> <filename at st> <filename at lt> <temp>')
        quit()

    filename1 = argv[1]
    filename2 = argv[2]
    T         = float(argv[3])

    data1 = torch.load(filename1)
    data2 = torch.load(filename2)

    tau_short = data1['tau_short']
    tau_long = data2['tau_short']
    boxsize  = data2['boxsize']

    max_ts_cut = 100
    pair_time_step = 0.1
    L_h = boxsize/2.
    q_max = math.sqrt(L_h*L_h + L_h*L_h)
    print('Maximum distance r = {}, r^2 = {}'.format(q_max,q_max*q_max))

    traj_st = data1['qp_trajectory']
    traj_lt = data2['qp_trajectory']
    # shape = [nsamples, (q,p), trajectory, nparticles, DIM]

    nsamples, _, iterations, nparticle, _ = traj_st.shape

    max_ts = iterations -1 # omit initial state

    gs_q = traj_st[:,0,:,:,:]
    q    = traj_lt[:,0,:,:,:]
    # shape = [nsamples, trajectory, nparticles, DIM]

    gs_p = traj_st[:,1,:,:,:]
    p    = traj_lt[:,1,:,:,:]
    # shape = [nsamples, trajectory, nparticles, DIM]


    del_q = torch.pow((gs_q - q),2)
    del_p =  torch.pow((gs_p - p),2)
    # shape = [nsamples, trajectory, nparticles, DIM]

    print('del q', del_q.shape, 'del_p', del_p.shape)

    avg_del_q_particle = torch.sum(del_q, dim=2) / nparticle
    # shape = [nsamples, trajectory, DIM]

    # print('----avg_del_q_sample each component----')
    # print(avg_del_q_sample)
    # print(avg_del_q_sample.shape)

    avg_del_q_particle = torch.sum(avg_del_q_particle, dim=2)
    # shape = [nsamples, trajectory]

    print('del_q per particle', avg_del_q_particle.shape)

    avg_del_p_particle = torch.sum(del_p, dim=2) / nparticle
    # shape = [nsamples, trajectory, DIM]

    # print('----avg_del_p_sample each component----')
    # print(avg_del_p_sample)
    # print(avg_del_p_sample.shape)

    avg_del_p_particle = torch.sum(avg_del_p_particle , dim=2)
    # shape = [nsamples, trajectory]

    print('del_p per particle', avg_del_p_particle.shape)

    del_qp_particle = avg_del_q_particle + avg_del_p_particle
    # shape = [nsamples, trajectory]

    avg2_del_qp_particle_sample = torch.sum( torch.pow(del_qp_particle , 2), dim=0) / nsamples

    print(avg2_del_qp_particle_sample.shape)

    avg_del_q_particle_sample = torch.sum(avg_del_q_particle, dim=0) / nsamples
    # shape = [trajectory]

    # print('----avg_del_q_sample_particle sum components----')
    # print(avg_del_q_sample_particle)

    avg_del_p_particle_sample = torch.sum(avg_del_p_particle, dim=0) / nsamples
    # shape = [trajectory]

    # print('----avg_del_p_sample_particle sum components----')
    # print(avg_del_p_sample_particle)

    avg_del_qp_particle_sample = avg_del_q_particle_sample + avg_del_p_particle_sample
    print('----avg_del_qp_sample_particle -----')
    # print(avg_del_qp_particle_sample)
    print(avg_del_qp_particle_sample.shape)

    std_del_qp_particle_sample = (avg2_del_qp_particle_sample - avg_del_qp_particle_sample*avg_del_qp_particle_sample)**0.5
    print('----std_del_qp_particle_sample -----')
    print(std_del_qp_particle_sample.shape)

    fig = plt.figure()
    t = torch.arange(0., max_ts_cut)
    plt.suptitle(r'nparticle {}, boxsize {:.4f}, Maximum distance $r = {:.4f}, r^2 = {:.4f}$'.format(nparticle,boxsize,q_max,q_max*q_max)
                 + '\nTemp = {}, pair with time step = {}, Maximum time step = {}'.format(T, pair_time_step, max_ts_cut)
                 + '\n' + r'gold standard $\tau^\prime$ = {}, time step $\tau$ = {} compared with gold standard'.format(tau_short, tau_long))

    fig.add_subplot(2,2,1)
    plt.ylabel(r'$\Delta q^{\tau,{\tau}^\prime}$',fontsize=15)
    # plt.ylim(-0.01,1e-10)
    plt.plot(t,avg_del_q_particle_sample[:max_ts_cut].detach().numpy(), label = 'Distance metric')
    # plt.xscale('log',base=2)
    # plt.yscale('log',base=2)

    fig.add_subplot(2,2,2)
    plt.ylabel(r'$\Delta p^{\tau,{\tau}^\prime}$',fontsize=15)
    plt.plot(t,avg_del_p_particle_sample[:max_ts_cut].detach().numpy(), label = 'Distance metric')
    # plt.xscale('log',base=2)
    # plt.yscale('log',base=2)

    fig.add_subplot(2,1,2)
    plt.xlabel('time',fontsize=16)
    plt.ylabel(r'$\Delta^{\tau,{\tau}^\prime}$',fontsize=18)
    plt.plot(t,avg_del_qp_particle_sample[:max_ts_cut].detach().numpy(), label = 'Distance metric')
    # plt.xscale('log',base=2)
    # plt.yscale('log',base=2)
    plt.tick_params(axis='y',labelsize=16)
    plt.show()
