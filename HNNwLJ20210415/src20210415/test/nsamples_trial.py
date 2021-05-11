import torch
import random

if __name__ == "__main__":

    random.seed(1)
    torch.manual_seed(1)

    n = 3
    m = 2
    a = torch.rand((n,m,2)) # shape is [3,2,2]
    print('e.g nsamples',n, 'nparticle',m, 'DIM 2' )
    print('a', a, '\n')

    print("not use for loop. this method takes same index randomly every n samples ")
    trial = torch.randint(0, a.shape[1], (1,))
    print('trial', trial)
    print(a[:,trial])

    print("use for loop. this method takes index randomly each sample")
    for i in range(n):

        trial = random.randint(0, a.shape[1] - 1)
        print('trial', trial)
        print(a[i, trial, :])