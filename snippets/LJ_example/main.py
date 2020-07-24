from phase_space   import phase_space
from pb            import periodic_bc
from LJ_term       import LJ_term

import numpy as np

if __name__=='__main__':

    p_list = np.array([[[.4,-.2],[.3, .2]]])
    q_list = np.array([[[.1,0.2],[.2,-.1]]])

    state = phase_space()
    state.set_p(p_list)
    state.set_q(q_list)

    bc = periodic_bc()

    eps = 1
    sig = 1
    L   = 2
    LJ06 = LJ_term(eps,sig, 6,L)
    LJ12 = LJ_term(eps,sig,12,L)

    energy = LJ6.energy(state,bc)
    dq = LJ6.evaluate_derivative_q(state,bc)


