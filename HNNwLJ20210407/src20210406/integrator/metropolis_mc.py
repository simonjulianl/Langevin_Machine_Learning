#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import copy
import random
import numpy as np
import time
import math
from parameters.MC_parameters import MC_parameters

class metropolis_mc:

    ''' This is a Monte Carlo Simulation only used to generate initial positions and sample equilibrium states'''

    _obj_count = 0

    def __init__(self):

        metropolis_mc._obj_count += 1
        assert (metropolis_mc._obj_count == 1), type(self).__name__ + " has more than one object"

        self.boxsize =MC_parameters.boxsize

    def position_sampler(self):

        ''' function to create random particle positions that are always between -0.5 * boxsize and 0.5 * boxsize

        return : torch.tensor
        shape is [1, nparticle, DIM]
        '''

        pos = np.random.uniform(-0.5, 0.5, (MC_parameters.nparticle, MC_parameters.DIM))
        pos = pos * self.boxsize
        pos = np.expand_dims(pos, axis=0)

        return torch.tensor(pos)

    def momentum_dummy_sampler(self):

        ''' function to make momentum zeros because not use for mc simulation

        return : torch.tensor
        shape is [1, nparticle, DIM]
        '''

        momentum = torch.zeros(MC_parameters.nparticle, MC_parameters.DIM)
        momentum = torch.unsqueeze(momentum, dim=0)

        return momentum

    def mcmove(self, hamiltonian, phase_space) :

        ''' MC method. if accepted, move to the new state, but if rejected, remain in the old state.

        parameter
        ------------
        curr_q : shape is [1, npaticle, DIM]
        dq     : float
                At low temperature, mostly reject not update new energy from Boltzmann factor.
                mulitiply displacement to increase acceptance rate

        enn_q  : update potential energy
        eno_q  : old potential energy
        '''

        curr_q = phase_space.get_q()

        self.eno_q = hamiltonian.total_energy(phase_space)

        trial = random.randint(0, curr_q.shape[1] - 1)           # randomly pick one particle from the state

        old_q = curr_q[:,trial,:].clone()

        # perform random step with proposed uniform distribution
        # if not move 0.5 , give the particle only a positive displacement
        curr_q[:, trial] = old_q + (torch.rand(1, MC_parameters.DIM) - 0.5) * self.boxsize * MC_parameters.dq

        phase_space.adjust_real(curr_q, self.boxsize)
        phase_space.set_q(curr_q)

        self.enn_q = hamiltonian.total_energy(phase_space)
        dU = self.enn_q - self.eno_q

        # accept with probability proportional di e ^ -beta * delta E
        self.ACCsum += 1.0
        self.ACCNsum += 1.0

        if (dU > 0):
            if (torch.rand([]) > math.exp( -dU / MC_parameters.temperature )):
                self.ACCsum -= 1.0      # rejected
                curr_q[:,trial] = old_q # restore the old position
                self.enn_q = self.eno_q
                phase_space.set_q(curr_q)


    def step(self, hamiltonian, phase_space):

        ''' Implementation of integration for Monte Carlo simulation

        parameter
        ___________
        phase space : contains q_list, p_list as input and contains boxsize also
        DISCARD     : discard initial mc steps
        niter       : iteration after discarded mc step
        nsamples    : the number of samples for mc

        Returns
        ___________
        q_list      : shape is [nsample, nparticle, DIM]
        U           : potential energy; shape is [nsamples, niter]
        AccRatio    : acceptance rate to update new energy ; shape is [nsamples]
        spec        : estimate specific heat from fluctuations of the potential energy; shape is [nsamples]

       '''

        niter = MC_parameters.iterations - MC_parameters.DISCARD
        q_list = torch.zeros((MC_parameters.nsamples, niter, MC_parameters.nparticle, MC_parameters.DIM), dtype = torch.float64)
        U = torch.zeros(MC_parameters.nsamples, niter)
        ACCRatio = torch.zeros(MC_parameters.nsamples)
        spec = torch.zeros(MC_parameters.nsamples)

        for z in range(0, MC_parameters.nsamples):

            self.ACCsum = 0.
            self.ACCNsum = 0.

            TE1sum = 0.0
            TE2sum = 0.0
            Nsum = 0.0

            phase_space.set_q(self.position_sampler())
            phase_space.set_p(self.momentum_dummy_sampler())
            phase_space.set_boxsize(self.boxsize)

            start = time.time()

            for i in range(0, MC_parameters.iterations):

                for _ in range(MC_parameters.DIM):
                    self.mcmove(hamiltonian, phase_space)

                if(i >= MC_parameters.DISCARD):

                    q_list[z,i- MC_parameters.DISCARD] = copy.deepcopy(phase_space.get_q())

                    if self.enn_q > MC_parameters.nparticle * MC_parameters.max_energy:

                        print('potential energy too high')
                        quit()

                    U[z,i- MC_parameters.DISCARD] = self.enn_q
                    TE1sum += self.enn_q
                    TE2sum += (self.enn_q * self.enn_q)
                    Nsum += 1.0

            ACCRatio[z] = self.ACCsum / self.ACCNsum
            spec[z] = (TE2sum / Nsum - TE1sum * TE1sum / Nsum / Nsum) / MC_parameters.temperature / MC_parameters.temperature / MC_parameters.nparticle

            end = time.time()

            print('finished taking {} configuration, '.format(z), 'temp: ', MC_parameters.temperature, 'Accratio :', ACCRatio[z], 'spec :', spec[z], 'dq: ', MC_parameters.dq, 'Discard: ', MC_parameters.DISCARD, 'time: ', end-start)

        #print out the rejection rate, recommended rejection 40 - 60 % based on Lit
        return q_list, U, ACCRatio, spec
