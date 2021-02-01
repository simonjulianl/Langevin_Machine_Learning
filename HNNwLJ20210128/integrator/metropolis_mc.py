#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from tqdm import trange
import copy
import random
import numpy as np
import warnings
from HNNwLJ20210128.parameters.MC_paramaters import MC_parameters

import math

class metropolis_mc:

    def __init__(self, noML_hamiltonian, phase_space):

        # seed setting
        try :
            seed = MC_parameters.seed
            torch.manual_seed(seed)
            # torch.backends.cudnn.deterministic = True  # Processing speed may be lower then when the models functions nondeterministically.
            # torch.backends.cudnn.benchmark = False
            torch.cuda.manual_seed_all(seed)

        except :
            warnings.warn('Seed not set, start using default numpy/random/torch seed')

        self.noML_hamiltonian = noML_hamiltonian
        self.phase_space = phase_space

    def position_sampler(self):

        pos = np.random.uniform(-0.5, 0.5,(MC_parameters.nparticle, MC_parameters.DIM))  ### ADD particle
        pos = pos * MC_parameters.boxsize
        pos = np.expand_dims(pos, axis=0)

        return torch.tensor(pos)

    def momentum_dummy_sampler(self):

        momentum = torch.zeros(MC_parameters.nparticle, MC_parameters.DIM)
        momentum = torch.unsqueeze(momentum, dim=0)

        return momentum

    def mcmove(self) :

        curr_q = self.phase_space.get_q()
        # print('curr_q', curr_q)
        self.eno_q = self.noML_hamiltonian.total_energy(self.phase_space)

        trial = random.randint(0, curr_q.shape[1] - 1) # randomly pick one particle from the state

        old_q = curr_q[:,trial].clone()
        # print('old_q', old_q)
        #perform random step with proposed uniform distribution
        curr_q[:,trial] = old_q + (torch.rand(1, MC_parameters.DIM)-0.5) * MC_parameters.dq

        self.phase_space.adjust_real(curr_q, MC_parameters.boxsize)
        self.phase_space.set_q(curr_q)

        self.enn_q = self.noML_hamiltonian.total_energy(self.phase_space)
        # print('enn_q when dU < 0', self.enn_q)

        dU = self.enn_q - self.eno_q
        self.phase_space.set_q(curr_q)

        #accept with probability proportional di e ^ -beta * delta E
        self.ACCsum += 1.0
        self.ACCNsum += 1.0
        if (dU > 0):
            if (torch.rand([]) > math.exp( -dU / MC_parameters.temperature )):
                self.ACCsum -= 1.0 # rejected
                curr_q[:,trial] = old_q # restore the old position

                self.enn_q = self.eno_q
                # print('enn_q when dU > 0', self.enn_q )
                self.phase_space.set_q(curr_q)


    def integrate(self):

        self.ACCsum = 0.
        self.ACCNsum = 0.
        # specific heat calc
        TE1sum = 0.0
        TE2sum = 0.0
        Nsum = 0.0
        MC_iterations = MC_parameters.iterations - MC_parameters.DISCARD
        q_list = torch.zeros((MC_iterations, MC_parameters.nparticle, MC_parameters.DIM), dtype = torch.float64)
        U = torch.zeros(MC_iterations)

        self.phase_space.set_q(self.position_sampler())
        self.phase_space.set_p(self.momentum_dummy_sampler())
        # print('q_list', self.phase_space.get_q())

        #for i in trange(0, self._state['iterations'], desc = "simulating"):
        for i in trange(0, MC_parameters.iterations):
            for _ in range(MC_parameters.DIM):
                self.mcmove()

            if(i >= MC_parameters.DISCARD):

                q_list[i- MC_parameters.DISCARD] = copy.deepcopy(self.phase_space.get_q())
                # print('save enn_q',self.enn_q)
                U[i - MC_parameters.DISCARD] = self.enn_q
                TE1sum += self.enn_q
                TE2sum += (self.enn_q * self.enn_q)
                Nsum += 1.0

        print('u', U)
        spec = (TE2sum / Nsum - TE1sum * TE1sum / Nsum / Nsum) / MC_parameters.temperature / MC_parameters.temperature  / MC_parameters.nparticle

        #print out the rejection rate, recommended rejection 40 - 60 % based on Lit
        
        return q_list, U, self.ACCsum/ self.ACCNsum, spec
    
    def __repr__(self):
        state = super().__repr__() 
        state += '\nIntegration Setting : \n'
        for key, value in self._state.items():
            state += str(key) + ': ' +  str(value) + '\n' 
        return state

