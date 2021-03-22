#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def linear_velocity_verlet(hamiltonian, phase_space, tau_cur, boxsize):
    '''
    velocity verlet integrator method

    Parameters
    ----------
    **state : dict
        all the configuration describing the state
        require :
            -Hamiltonian : hamiltonian
                list of functions modelling the energy
            -m : float
                mass
            -q : np.array of N X DIM
            -p : np.array of N X DIM

    Returns
    -------
    state : dict
        updated configuration state after 1 integration method

    '''
    # get all the constants
    q = phase_space.get_q()
    p = phase_space.get_p()

    tau = tau_cur

    # print('vv input, tau', q, p, tau)

    # p_list_dummy = np.zeros(p.shape)  # to prevent KE from being integrated
    # state['phase_space'].set_p(p_list_dummy)

    p = p + tau / 2 * (-hamiltonian.dHdq(phase_space))  # dp/dt
    phase_space.set_p(p)

    # print('update p', p)

    q = q + tau * p  # dq/dt = dK/dp = p, q is not in dimensionless unit
    # print('before adjust q',q)

    phase_space.adjust_real(q, boxsize) # enforce boundary condition - put particle back into box
    phase_space.set_q(q)

    # print('update q', q)

    # if __debug__:
    #     phase_space.debug_pbc(q, boxsize)

    p = p + tau / 2 * (-hamiltonian.dHdq(phase_space))  # dp/dt

    # print('update p', p)
    # print('update q, update p', q, p)

    phase_space.set_q(q) # dimensionless in hamiltonian.dHdq(phase_space) so that need one more time
    phase_space.set_p(p)  # update state after 1 step

    return q, p


linear_velocity_verlet.name = 'linear_velocity_verlet'  # add attribute to the function for marker
