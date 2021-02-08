#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from tqdm import trange
import copy
import random
import numpy as np
import warnings
from MC_paramaters import MC_parameters

import math

class metropolis_mc:

    _obj_count = 0

    def __init__(self):

        metropolis_mc._obj_count += 1
        assert (metropolis_mc._obj_count == 1), type(self).__name__ + " has more than one object"

        # seed setting
        try :
            seed = MC_parameters.seed
            torch.manual_seed(seed)
            # torch.backends.cudnn.deterministic = True  # Processing speed may be lower then when the models functions nondeterministically.
            # torch.backends.cudnn.benchmark = False
            torch.cuda.manual_seed_all(seed)

        except :
            warnings.warn('Seed not set, start using default numpy/random/torch seed')


    def position_sampler(self):

        pos = np.random.uniform(-0.5, 0.5,(MC_parameters.nparticle, MC_parameters.DIM))  ### ADD particle
        pos = pos * MC_parameters.boxsize
        pos = np.expand_dims(pos, axis=0)

        return torch.tensor(pos)

    def momentum_dummy_sampler(self):

        momentum = torch.zeros(MC_parameters.nparticle, MC_parameters.DIM)
        momentum = torch.unsqueeze(momentum, dim=0)

        return momentum

    def mcmove(self, hamiltonian, phase_space) :

        curr_q = phase_space.get_q()
        # print('curr_q', curr_q)
        self.eno_q = hamiltonian.total_energy(phase_space)

        trial = random.randint(0, curr_q.shape[1] - 1) # randomly pick one particle from the state

        old_q = curr_q[:,trial].clone()
        # print('old_q', old_q)
        #perform random step with proposed uniform distribution
        #curr_q[:,trial] = old_q + (torch.rand(1, MC_parameters.DIM)-0.5) * MC_parameters.dq -- wrong
        curr_q[:, trial] = old_q + (torch.rand(1, MC_parameters.DIM) - 0.5) * MC_parameters.boxsize * MC_parameters.dq

        phase_space.adjust_real(curr_q, MC_parameters.boxsize)
        phase_space.set_q(curr_q)

        self.enn_q = hamiltonian.total_energy(phase_space)
        # print('enn_q when dU < 0', self.enn_q)

        dU = self.enn_q - self.eno_q
        phase_space.set_q(curr_q)

        #accept with probability proportional di e ^ -beta * delta E
        self.ACCsum += 1.0
        self.ACCNsum += 1.0
        if (dU > 0):
            if (torch.rand([]) > math.exp( -dU / MC_parameters.temperature )):
                self.ACCsum -= 1.0 # rejected
                curr_q[:,trial] = old_q # restore the old position

                self.enn_q = self.eno_q
                # print('enn_q when dU > 0', self.enn_q )
                phase_space.set_q(curr_q)


    def step(self, hamiltonian, phase_space):

        self.ACCsum = 0.
        self.ACCNsum = 0.
        # specific heat calc
        TE1sum = 0.0
        TE2sum = 0.0
        Nsum = 0.0

        MC_iterations = MC_parameters.iterations - MC_parameters.DISCARD
        q_list = torch.zeros((MC_iterations, MC_parameters.nparticle, MC_parameters.DIM), dtype = torch.float64)
        U = torch.zeros(MC_iterations)

        phase_space.set_q(self.position_sampler())
        phase_space.set_p(self.momentum_dummy_sampler())
        print('q, p for mcs', self.position_sampler(), self.momentum_dummy_sampler())

        #for i in trange(0, self._state['iterations'], desc = "simulating"):
        for i in trange(0, MC_parameters.iterations):
            for _ in range(MC_parameters.DIM):
                self.mcmove(hamiltonian, phase_space)

            if(i >= MC_parameters.DISCARD):

                q_list[i- MC_parameters.DISCARD] = copy.deepcopy(phase_space.get_q())
                # print('save enn_q',self.enn_q)

                if self.enn_q > MC_parameters.nparticle * 10**3:

                    print('potential energy too high')
                    quit()

                U[i - MC_parameters.DISCARD] = self.enn_q
                TE1sum += self.enn_q
                TE2sum += (self.enn_q * self.enn_q)
                Nsum += 1.0

        # print('u', U)
        spec = (TE2sum / Nsum - TE1sum * TE1sum / Nsum / Nsum) / MC_parameters.temperature / MC_parameters.temperature  / MC_parameters.nparticle

        print('Accratio', self.ACCsum/ self.ACCNsum, spec)
        #print out the rejection rate, recommended rejection 40 - 60 % based on Lit
        
        return q_list, U, self.ACCsum/ self.ACCNsum, spec
    
    def __repr__(self):
        state = super().__repr__() 
        state += '\nIntegration Setting : \n'
        for key, value in self._state.items():
            state += str(key) + ': ' +  str(value) + '\n' 
        return state

