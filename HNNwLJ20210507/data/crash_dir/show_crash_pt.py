import sys
import torch

if __name__ == '__main__':
    # print qp list before crash

    argv = sys.argv
    filename = argv[1]

    data = torch.load(filename)
    # shape is [(q,p), crashed nsamples, npaticle, DIM]

    print('print qp list', data)
    print('print qp shape', data['qp_trajectory'].shape)
