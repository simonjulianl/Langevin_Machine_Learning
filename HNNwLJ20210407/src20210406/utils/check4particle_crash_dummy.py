import torch
from datetime import datetime


class check4particle_crash_dummy:

    def __init__(self, backward_method, ethrsh, pthrsh, boxsize):
        '''
        this class do nothing, use for case when we want to switch off check
        '''
        return


    def check(self,phase_space,hamiltonian,tau):
        # do no check and return
        return

