#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np 

def linear_velocity_verlet(Hamiltonian, **state) :

    #get all the constants
    q = state['phase_space'].get_q()
    p = state['phase_space'].get_p()

    tau = state['tau']
    pb = state['pb_q']
    boxsize = state['BoxSize']

    print('vv',q,p,tau)

    # p_list_dummy = np.zeros(p.shape) # to prevent KE from being integrated
    # state['phase_space'].set_p(p_list_dummy)

    p = p + tau / 2 * ( -Hamiltonian.dHdq(state['phase_space'], pb) ) #dp/dt
    print('cor',-Hamiltonian.dHdq(state['phase_space'], pb) )
    print('vv p 1 update',q,p,tau)

    state['phase_space'].set_p(p)

    q = q + tau * p # dq/dt = dK/dp = p

    pb.adjust_real(q, boxsize)
    state['phase_space'].set_q(q)

    pb.debug_pbc(q, boxsize)

    print('vv q update',q,p,tau)

    p = p + tau / 2 * ( -Hamiltonian.dHdq(state['phase_space'], pb) ) #dp/dt

    state['phase_space'].set_q(q) ; state['phase_space'].set_p(p) # update state after 1 step
    print('vv p 2 update',q,p,tau)
    return state 

linear_velocity_verlet.name = 'linear_velocity_verlet' # add attribute to the function for marker
