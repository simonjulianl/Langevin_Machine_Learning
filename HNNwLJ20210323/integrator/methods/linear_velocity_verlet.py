#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch

def linear_velocity_verlet(hamiltonian, phase_space, tau_cur, boxsize):
    '''
    velocity verlet integrator method

    parameter
    -------
    tau : torch.tensor
            duplicate tau_cur when load batch more than one
    Returns
    -------
    state : q, p
    '''

    q = phase_space.get_q()
    p = phase_space.get_p()

    tau_cur = torch.unsqueeze(tau_cur,dim=1)
    tau_cur = torch.repeat_interleave(tau_cur, q.shape[1] * q.shape[2])
    tau = tau_cur.reshape(-1,q.shape[1], q.shape[2])

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
