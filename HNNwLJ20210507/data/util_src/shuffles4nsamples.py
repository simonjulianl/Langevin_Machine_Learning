import sys
import torch

from extract2files import save_to

if __name__ == "__main__":
    # run like this
    # python shuffles4nsamples.py ../gen_by_MC/train/n4Tallnsamples.pt ../gen_by_MC/train/n4Tallnsamples36000_shuffle.pt
    torch.manual_seed(2145)
  
    argv = sys.argv

    if len(argv) != 3:
        print('usage <programe> <old filename> <new filename>')
        quit()

    pathname  = argv[1]
    outfile   = argv[2] # new pathname

    data      = torch.load(pathname)

    qp_list   = data['qp_trajectory']
    # shape is [nsamples, (q,p), 1, nparticles, DIM]

    tau_short = data['tau_short']
    tau_long  = data['tau_long'] 
    boxsize   = data['boxsize']

    idx = torch.randperm(qp_list.shape[0])

    qp_shuffle = qp_list[idx]

    save_to(outfile, qp_shuffle, tau_short, tau_long, boxsize)
    print('write pathname', outfile)

