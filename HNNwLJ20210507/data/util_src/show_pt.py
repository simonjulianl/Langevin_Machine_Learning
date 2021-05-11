import sys
import torch

if __name__ == '__main__':
    # make sure that multiple components correct or not
    argv = sys.argv
    filename = argv[1]

    data = torch.load(filename)

    qp_trajectory = data['qp_trajectory']
    tau_short     = data['tau_short']
    tau_long      = data['tau_long']
    boxsize       = data['boxsize']

    print('print qp list', qp_trajectory)
    print('qp shape ', qp_trajectory.shape, 'tau short ', tau_short, 'tau long ', tau_long, 'boxsize', boxsize)
