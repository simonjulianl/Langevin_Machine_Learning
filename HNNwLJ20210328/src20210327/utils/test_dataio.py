import torch
from data_io import data_io

def generate_randn(nparticle, DIM):
    ''' put random points onto
        2 : q, p '''
    qp_list = torch.randn(2, nparticle, DIM)

    return qp_list

def generate_qp_mc(nsamples, nparticle, DIM):
    ''' shape is [(q,p), nsamples, nparticle, DIM]

    return
    torch.tensor
    '''

    init_qp = []

    for i in range(nsamples):
        qp_list = generate_randn(nparticle, DIM)
        init_qp.append(qp_list)

    init_qp_ = torch.stack(init_qp)
    init_qp_ = init_qp_.permute(1,0,2,3)

    return init_qp_
   

def generate_trajectory_qp(nsamples, nparticle, DIM, nsteps):
    ''' shape is [nstep, nsamples, (q,p), nparticle, DIM]

    return
    torch.tensor
    '''

    qp_traj = []
    for t in range(nsteps):
        qp_step = generate_qp_mc(nsamples, nparticle, DIM)
        qp_traj.append(qp_step)

    qp_traj_ = torch.stack(qp_traj)
    return qp_traj_
    

if __name__=='__main__':

    root_dir_name = './'
    io = data_io(root_dir_name)

    nsamples = 3
    nparticle = 2
    DIM = 2
    nsteps = 4

    # testing io for init_qp
    init_qp_filename = 'initqp.pt'

    init_qp = generate_qp_mc(nsamples, nparticle, DIM)
    io.write_init_qp(init_qp_filename, init_qp)

    qp_readback = io.read_init_qp(init_qp_filename)

    print(qp_readback.shape)
    # tensor_qp_list = torch.stack(qp_readback)

    # compare if init_qp == qp_readback
    assert torch.all(torch.eq(init_qp, qp_readback))

    for t in range(4):
        # testing io for trajectories_qp
        trajectory_qp_filename = 'trajectories_qp'+str(t)+'.pt'
        
        qp_trajectory = generate_trajectory_qp(nsamples, nparticle, DIM, nsteps)
        
        io.write_trajectory_qp(trajectory_qp_filename, qp_trajectory)
        
        qp_trajectory_readback = io.read_trajectory_qp(trajectory_qp_filename)
        
        assert torch.all(torch.eq(qp_trajectory, qp_trajectory_readback))
        
        print(qp_trajectory_readback.shape)
