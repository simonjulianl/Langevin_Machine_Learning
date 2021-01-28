from momentum_sampler import momentum_sampler

if __name__ == '__main__' :

    nsamples = 4
    nparticle = 2
    DIM = 2
    epsilon = 1.
    sigma = 1.
    mass = 1
    T = 0.04

    state = {
        'nsamples': nsamples,
        'temperature' : T,
        'nparticle': nparticle,
        'm' : mass
    }

    momentum_sampler = momentum_sampler(**state)
    p_list = momentum_sampler.momentum_samples()
    print(p_list)