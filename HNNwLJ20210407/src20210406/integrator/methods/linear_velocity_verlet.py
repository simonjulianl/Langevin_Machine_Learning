#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def linear_velocity_verlet(hamiltonian, phase_space, tau_cur, boxsize):
    '''
    velocity verlet integrator method

    Parameters
    ----------
    hamiltonian     : can be ML or noML hamiltonian
    phase_space     : real-space-unit, contains q_list, p_list as input for integration
    tau_cur         : float
    boxsize         : float

    Returns
    -------
    state :
    q shape is [nsamples, nparticle, DIM]
    p shape is [nsamples, nparticle, DIM]
    '''

    q = phase_space.get_q()
    p = phase_space.get_p()

    tau = tau_cur

    p = p + tau / 2 * (-hamiltonian.dHdq1(phase_space))  # dp/dt
    phase_space.set_p(p)

    q = q + tau * p  # dq/dt = dK/dp = p, q is not in dimensionless unit

    phase_space.adjust_real(q, boxsize) # enforce boundary condition - put particle back into box
    phase_space.set_q(q)

    p = p + tau / 2 * (-hamiltonian.dHdq2(phase_space))  # dp/dt

    phase_space.set_p(p)  # update state after 1 step

    return q, p

def linear_velocity_verlet_backward(hamiltonian, phase_space, tau_cur, boxsize):
    '''
    backward velocity verlet integrator method

    Returns
    -------
    state :
    q shape is [nsamples, nparticle, DIM]
    p shape is [nsamples, nparticle, DIM]
    '''

    q = phase_space.get_q()
    p = phase_space.get_p()

    tau = tau_cur

    p = p + (tau / 2) * (hamiltonian.dHdq1(phase_space))  # dp/dt
    phase_space.set_p(p)

    q = q - tau * p  # dq/dt = dK/dp = p, q is not in dimensionless unit

    phase_space.adjust_real(q, boxsize) # enforce boundary condition - put particle back into box
    phase_space.set_q(q)

    p = p + (tau / 2) * (hamiltonian.dHdq2(phase_space))  # dp/dt

    phase_space.set_p(p)  # update state after 1 step

    return q, p

linear_velocity_verlet.name = 'linear_velocity_verlet'  # add attribute to the function for marker
