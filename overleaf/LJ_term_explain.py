import numpy as np
from Interaction import Interaction

class LJ_term(Interaction):
    def __init__(self, epsilon : float, sigma : float, exponent : float, boxsize : float):
        '''
        Parameters
        ----------
        epsilon : float
            depth of potential well
        sigma : float
            finite distance at which the inter-particle potential is zero
        '''
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
        Function to calculate dHdxi
        
        Returns
        -------
        dphidxi: np.array 
            dphidxi calculated given the terms of N X N_particle X DIM 

        '''
        xi_state = phase_space.get_q()
        p_state  = phase_space.get_p()
        # N:The num of samples; N_particle:The num of particles; DIM: Dimension
        N, N_particle,DIM  = xi_state.shape  
        # derivative of separable term in N X N_particle X DIM matrix
        dphidxi = np.zeros(xi_state.shape) 

        for k in range(N):
            pb.adjust(xi_state[k])
            # delta_xi = [[xi_{k_1}-xi_{j_1},xi_{k_2}-xi_{j_2}],[xi_{j_1}-xi_{k_1},xi_{j_2}-xi_{k_2}]]
            # q=dd=xi=[[0, |xi_k - xi_j|],[|xi_k - xi_j|,0]]
            delta_xi, q = pb.paired_distance(xi_state[k])
            dphidq = eval(self._derivative_q)   # -6.0*q**(-7.0) = -6.0*xi**(-7.0)
            dphidq[~np.isfinite(dphidq)] = 0
            # dphidxi = [[dphi/xi_{k_1},dph/xi_{k_2}],[-dphi/xi_{k_1},-dph/xi_{k_2}]]
            dphidxi[k] = np.sum(dphidq,axis=1) * delta_xi / np.sum(q,axis=1)  

        dphidxi = dphidxi * (parameter_term / self._boxsize )
        return dphidxi

