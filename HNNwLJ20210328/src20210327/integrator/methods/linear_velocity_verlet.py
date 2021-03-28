#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def linear_velocity_verlet(hamiltonian, phase_space, tau_cur, boxsize):
    '''
    velocity verlet integrator method

    Parameters
    ----------
    tau : float
    boxsize : float

    Returns
    -------
    state :
    q shape is [nsamples, nparticle, DIM]
    p shape is [nsamples, nparticle, DIM]
    '''

    q = phase_space.get_q()
    p = phase_space.get_p()

    tau = tau_cur

    p = p + tau / 2 * (-hamiltonian.dHdq(phase_space))  # dp/dt
    phase_space.set_p(p)

    q = q + tau * p  # dq/dt = dK/dp = p, q is not in dimensionless unit

    phase_space.adjust_real(q, boxsize) # enforce boundary condition - put particle back into box
    phase_space.set_q(q)

    # if __debug__:
    #     phase_space.debug_pbc(q, boxsize)

    p = p + tau / 2 * (-hamiltonian.dHdq(phase_space))  # dp/dt

    phase_space.set_q(q) # dimensionless in hamiltonian.dHdq(phase_space) so that need one more time
    phase_space.set_p(p)  # update state after 1 step

    return q, p


linear_velocity_verlet.name = 'linear_velocity_verlet'  # add attribute to the function for marker
