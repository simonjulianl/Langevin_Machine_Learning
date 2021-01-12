#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from tqdm import trange
import copy
import random
import numpy as np
import warnings
import math

class metropolis_mc:

    def __init__(self, noML_hamiltonian, **state):

        # seed setting
        try :
            seed = state['seed']
            torch.manual_seed(seed)
            # torch.backends.cudnn.deterministic = True  # Processing speed may be lower then when the model functions nondeterministically.
            # torch.backends.cudnn.benchmark = False
            torch.cuda.manual_seed_all(seed)

        except :
            warnings.warn('Seed not set, start using default numpy/random/torch seed')

        self.noML_hamiltonian = noML_hamiltonian
        self._state = state

    def position_sampler(self):

        pos = np.random.uniform(-0.5, 0.5,(self._state['nparticle'], self._state['DIM']))  ### ADD particle
        pos = pos * self._state['boxsize']
        pos = np.expand_dims(pos, axis=0)

        return torch.tensor(pos)

    def momentum_dummy_sampler(self):

        return torch.zeros(self._state['nsamples'],self._state['nparticle'],self._state['DIM'])

    def mcmove(self) :

        curr_q = self._state['phase_space'].get_q()
        self.eno_q = self.noML_hamiltonian.total_energy(self._state['phase_space'],self._state['pb_q'])

        trial = random.randint(0, curr_q.shape[1]-1) # randomly pick one particle from the state

        old_q = curr_q[:,trial].clone()
        #perform random step with proposed uniform distribution
        curr_q[:,trial] = old_q + (torch.rand(1,self._state['DIM'])-0.5)* self._state['dq']

        self._state['pb_q'].adjust_real(curr_q,self._state['boxsize'])
        self._state['phase_space'].set_q(curr_q)

        self.enn_q = self.noML_hamiltonian.total_energy(self._state['phase_space'],self._state['pb_q'])

        dU = self.enn_q - self.eno_q
        self._state['phase_space'].set_q(curr_q)

        #accept with probability proportional di e ^ -beta * delta E
        self.ACCsum += 1.0
        self.ACCNsum += 1.0
        if (dU > 0):
            if (torch.rand([]) > math.exp( -dU / self._state['temperature'] )):
                self.ACCsum -= 1.0 # rejected
                curr_q[:,trial] = old_q # restore the old position
                self.enn_q = self.eno_q
                self._state['phase_space'].set_q(curr_q)
        
        
    def integrate(self):

        self.ACCsum = 0.
        self.ACCNsum = 0.
        # specific heat calc
        TE1sum = 0.0
        TE2sum = 0.0
        Nsum = 0.0
        iterations = self._state['iterations'] - self._state['DISCARD']
        q_list = torch.zeros((iterations,self._state['nparticle'],self._state['DIM']),dtype = torch.float64)
        U = torch.zeros(iterations)

        self._state['phase_space'].set_q(self.position_sampler())
        self._state['phase_space'].set_p(self.momentum_dummy_sampler())
        # print('q_list', self._state['phase_space'].get_q())

        for i in trange(0, self._state['iterations'], desc = "simulating"):
            for _ in range(self._state['DIM']):
                self.mcmove()
            if(i >= self._state['DISCARD']):
                q_list[i-self._state['DISCARD']] = copy.deepcopy(self._state['phase_space'].get_q())
                U[i-self._state['DISCARD']] = self.enn_q
                TE1sum += self.enn_q
                TE2sum += (self.enn_q * self.enn_q)
                Nsum += 1.0

        spec = (TE2sum / Nsum - TE1sum * TE1sum / Nsum / Nsum) / self._state['temperature'] / self._state['temperature']  / self._state['nparticle']

        #print out the rejection rate, recommended rejection 40 - 60 % based on Lit
        
        return q_list, U, self.ACCsum/ self.ACCNsum, spec
    
    def __repr__(self):
        state = super().__repr__() 
        state += '\nIntegration Setting : \n'
        for key, value in self._state.items():
            state += str(key) + ': ' +  str(value) + '\n' 
        return state

