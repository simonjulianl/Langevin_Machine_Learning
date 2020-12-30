#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from numpy import newaxis

def harmonic_velocity_verlet(tau,**state) :

    q = state['phase_space'].get_q()
    p = state['phase_space'].get_p()

    N, N_particle,DIM = q.shape # N (nsample) x N particle x DIM

    Hamiltonian = state['hamiltonian']

    p_list_dummy = np.zeros(p.shape)  # to prevent KE from being integrated

    # state['phase_space'].set_p(p_list_dummy)
    # HK for code checking only U = Hamiltonian.total_energy(state['phase_space'], state['pb_q'])

    # derivative of potential V
    state['phase_space'].set_q(q)
    state['phase_space'].set_p(p_list_dummy)
    V = Hamiltonian.dHdq(state['phase_space'], state['pb_q'])

    # second derivative of potential G
    state['phase_space'].set_q(q)
    state['phase_space'].set_p(p_list_dummy)
    G = Hamiltonian.d2Hdq2(state['phase_space'], state['pb_q'])

    u_ = np.zeros([N, N_particle * DIM, N_particle * DIM]) #eigenvectors of G terms of nsamples
    y_ = np.zeros([N, N_particle * DIM, 1])  # y terms of nsamples
    ydot_ = np.zeros([N, N_particle * DIM, 1]) # ydot terms of nsamples

    for z in range(N):

        vals, u = np.linalg.eigh(G[z])  # vals = each eigenvalue of G, that are mutually orthogonal

        vals_matrix = np.diag(vals) # eigenvalues of diagonal matrix

        inv_u = np.linalg.inv(u) # inverse of eigenvectors

        vals = np.expand_dims(vals, axis=1)

        kappa = 1e-6 # To avoid big numerical errors when eigenvalue is small

        # divide index for positive eigenvalue and negative eigenvalue
        index_zero = np.where(vals == 0)[0] # index when eigenavlue = 0
        index_positive_ae = np.where(vals > kappa)[0] # index when positive eigenvalue > kappa
        index_positive_be = np.where( (vals > 0) & (vals <= kappa))[0] # index when 0 < positive eigenvalue < kappa
        index_negative_be = np.where(vals < -kappa)[0]  # index when negative eigenvalue < -kappa
        index_negative_ae = np.where( (vals < 0) & (vals >= -kappa))[0] # index when -kappa < negative eigenvalue < 0

        vals_zero = vals[index_zero]  #  eigenavlue = 0
        vals_positive_ae = vals[index_positive_ae] # positive eigenvalue > kappa
        vals_positive_be = vals[index_positive_be] # 0 < positive eigenvalue < kappa
        vals_negative_be = vals[index_negative_be] # negative eigenvalue < -kappa
        vals_negative_ae = vals[index_negative_ae]  # -kappa < negative eigenvalue < 0

        grad_U = V[z].reshape(-1, 1)

        q0 = q[z].reshape(-1, 1)
        qdot0 = p[z].reshape(-1, 1)  # qdot0 = p0

        # initial condition
        y0 = np.dot(inv_u, q0) # y0 = u^T q0

        # initial condition
        ydot0 = np.dot(inv_u, qdot0) # ydot0 = u^T qdot0

        s = -np.dot(inv_u,grad_U) + np.dot(vals_matrix,np.dot(inv_u,q0)) # s = u^T( -V(q0)+G(q0) q0 )
        s_eigen_zero = -np.dot(inv_u,grad_U) # # s = u^T( -V(q0) ) when eigenvalue = 0

        y = np.zeros((y0[:newaxis] + tau).shape) # y = u^T q
        ydot = np.zeros((y0[:newaxis] + tau).shape)  # ydot = u^T qdot

        if (vals_zero == 0).all():
            y[index_zero] = 1./2. * s_eigen_zero[index_zero] * tau * tau + ydot0[index_zero] * tau + y0[index_zero]
            ydot[index_zero] = s_eigen_zero[index_zero] * tau + ydot0[index_zero]


        if vals_positive_ae.all():
            y[index_positive_ae] = (y0[index_positive_ae]-(s[index_positive_ae] / vals[index_positive_ae])) * np.cos(np.sqrt(vals[index_positive_ae]) * tau) + 1. / np.sqrt(
                vals[index_positive_ae]) * ydot0[index_positive_ae] * np.sin(np.sqrt(vals[index_positive_ae]) * tau) + (s[index_positive_ae] / vals[index_positive_ae])

            ydot[index_positive_ae] = -(y0[index_positive_ae]-(s[index_positive_ae] / vals[index_positive_ae])) * np.sqrt(vals[index_positive_ae]) * np.sin(
                np.sqrt(vals[index_positive_ae]) * tau) + ydot0[index_positive_ae] * np.cos(np.sqrt(vals[index_positive_ae]) * tau)


        if vals_positive_be.all():
            y[index_positive_be] = 1./2.* s_eigen_zero[index_positive_be] * tau * tau + ydot0[index_positive_be] * tau + y0[index_positive_be]

            ydot[index_positive_be] = s_eigen_zero[index_positive_be] * tau + ydot0[index_positive_be]


        if vals_negative_be.all():

            y[index_negative_be] = (y0[index_negative_be] * np.sqrt(np.abs(vals[index_negative_be])) + ydot0[index_negative_be] + (s[index_negative_be] / np.sqrt(np.abs(vals[index_negative_be])))) / (
                        2 * np.sqrt(np.abs(vals[index_negative_be]))) * np.exp(np.sqrt(np.abs(vals[index_negative_be])) * tau) + (
                                            y0[index_negative_be] * np.sqrt(np.abs(vals[index_negative_be])) - ydot0[index_negative_be] + (s[index_negative_be] / np.sqrt(np.abs(vals[index_negative_be])))) / (
                                            2 * np.sqrt(np.abs(vals[index_negative_be]))) * np.exp(-np.sqrt(np.abs(vals[index_negative_be])) * tau) - (s[index_negative_be]/np.abs(vals[index_negative_be]) )

            ydot[index_negative_be] = (y0[index_negative_be] * np.sqrt(np.abs(vals[index_negative_be])) + ydot0[index_negative_be]+ (s[index_negative_be] / np.sqrt(np.abs(vals[index_negative_be])))) / 2 * np.exp(
                np.sqrt(np.abs(vals[index_negative_be])) * tau) - (y0[index_negative_be] * np.sqrt(np.abs(vals[index_negative_be])) - ydot0[index_negative_be]+ (s[index_negative_be] / np.sqrt(np.abs(vals[index_negative_be])))) / 2 * np.exp(
                -np.sqrt(np.abs(vals[index_negative_be])) * tau)


        if vals_negative_ae.all():
            y[index_negative_ae] = 1. / 2. * s_eigen_zero[index_negative_ae] * tau * tau + ydot0[index_negative_ae] * tau + y0[
                index_negative_ae]

            ydot[index_negative_ae] = s_eigen_zero[index_negative_ae] * tau + ydot0[index_negative_ae]

        u_[z] = u
        y_[z] = y
        ydot_[z] = ydot

    return u_, y_, ydot_

harmonic_velocity_verlet.name = 'harmonic_velocity_verlet' # add attribute to the function for marker
