import numpy as np
from Interaction import Interaction

class LJ_term(Interaction):
    def __init__(self, epsilon : float, sigma : float, exponent : float, boxsize : float):

        try:
            self._epsilon  = float(epsilon)
            self._sigma    = float(sigma)
            self._boxsize  = float(boxsize)
            self._exponent = float(exponent)

        except :
            raise Exception('sigma / epsilon rror')

        super().__init__('1.0 / q ** {0} '.format(self._exponent))
        parameter_term = (4 * self._epsilon) * ((self._sigma / self._boxsize) ** self._exponent
        self._name = 'Lennard Jones Potential'


    def evaluate_derivative_q(self, phase_space, pb):
        '''
        dphidxi: np.array 
            dphidxi calculated given the terms of N X N_particle X DIM 
        '''
        xi_state = phase_space.get_q()
        p_state  = phase_space.get_p()
        # N:The num of samples; N_particle:The num of particles
        N, N_particle,DIM  = xi_state.shape  
        # derivative of separable term in N X N_particle X DIM matrix
        dphidxi = np.zeros(xi_state.shape) 

        for z in range(N):
            pb.adjust(xi_state[z])
            delta_xi, q = pb.paired_distance(xi_state[z])
            dphidq = eval(self._derivative_q)  
            dphidq[~np.isfinite(dphidq)] = 0
            dphidxi[z] = np.sum(dphidq,axis=1) * delta_xi / np.sum(q,axis=1)  

        dphidxi = dphidxi * (parameter_term / self._boxsize )
        return dphidxi

