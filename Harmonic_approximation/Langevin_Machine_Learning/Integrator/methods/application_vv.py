#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from numpy import newaxis

def application_vv(time_step,**state) :

    q = state['phase_space'].get_q()
    p = state['phase_space'].get_p()
    N, N_particle,DIM = q.shape

    Hamiltonian = state['hamiltonian']

    p_list_dummy = np.zeros(p.shape)  # to prevent KE from being integrated
    state['phase_space'].set_p(p_list_dummy)

    # HK for code checking only U = Hamiltonian.total_energy(state['phase_space'], state['pb_q'])

    state['phase_space'].set_q(q)
    state['phase_space'].set_p(p_list_dummy)
    V = Hamiltonian.dHdq(state['phase_space'], state['pb_q'])

    state['phase_space'].set_q(q)
    state['phase_space'].set_p(p_list_dummy)
    G = Hamiltonian.d2Hdq2(state['phase_space'], state['pb_q'])

    vecs_ = np.zeros([N,N_particle*DIM,N_particle*DIM])
    y_ = np.zeros([N,N_particle*DIM,1])
    y_diff_ = np.zeros([N,N_particle*DIM,1])

    for z in range(N):

        vals, vecs = np.linalg.eigh(G[z])

        vals_matrix = np.diag(vals)

        inv_vecs = np.linalg.inv(vecs)

        vals = np.expand_dims(vals, axis=1)

        epsilon = 1e-6
        # divide index for positive eigenvalue and negative eigenvalue
        index_zero = np.where(vals == 0)[0]
        index_positive_ae = np.where(vals > epsilon)[0] #above positive epsilon
        index_positive_be = np.where( (vals > 0) & (vals <= epsilon))[0] #below positive epsilon
        index_negative_be = np.where(vals < -epsilon)[0]  #below negative epsilon
        index_negative_ae = np.where( (vals < 0) & (vals >= -epsilon))[0] #above negative epsilon

        vals_zero = vals[index_zero]
        vals_positive_ae = vals[index_positive_ae]
        vals_positive_be = vals[index_positive_be]
        vals_negative_be = vals[index_negative_be]
        vals_negative_ae = vals[index_negative_ae]

        grad_potential = V[z].reshape(-1, 1)

        q0 = q[z].reshape(-1, 1)
        q0_diff = p[z].reshape(-1, 1)  # q'(0) = p

        # initial condition
        y0 = np.dot(inv_vecs, q0)

        # initial condition
        y0_diff = np.dot(inv_vecs, q0_diff)

        s = -np.dot(inv_vecs,grad_potential) + np.dot(vals_matrix,np.dot(inv_vecs,q0))
        s_lambda_zero = -np.dot(inv_vecs,grad_potential)
        t = time_step

        y = np.zeros((y0[:newaxis] + t).shape)
        y_diff = np.zeros((y0[:newaxis] + t).shape)

        if (vals_zero == 0).all():
            y[index_zero] = 1./2. * s_lambda_zero[index_zero] * t * t + y0_diff[index_zero] * t + y0[index_zero]
            y_diff[index_zero] = s_lambda_zero[index_zero] * t + y0_diff[index_zero]


        if vals_positive_ae.all():
            y[index_positive_ae] = (y0[index_positive_ae]-(s[index_positive_ae] / vals[index_positive_ae])) * np.cos(np.sqrt(vals[index_positive_ae]) * t) + 1. / np.sqrt(
                vals[index_positive_ae]) * y0_diff[index_positive_ae] * np.sin(np.sqrt(vals[index_positive_ae]) * t) + (s[index_positive_ae] / vals[index_positive_ae])

            y_diff[index_positive_ae] = -(y0[index_positive_ae]-(s[index_positive_ae] / vals[index_positive_ae])) * np.sqrt(vals[index_positive_ae]) * np.sin(
                np.sqrt(vals[index_positive_ae]) * t) + y0_diff[index_positive_ae] * np.cos(np.sqrt(vals[index_positive_ae]) * t)


        if vals_positive_be.all():
            y[index_positive_be] = 1./2.* s_lambda_zero[index_positive_be] * t * t + y0_diff[index_positive_be] * t + y0[index_positive_be]

            y_diff[index_positive_be] = s_lambda_zero[index_positive_be] * t + y0_diff[index_positive_be]


        if vals_negative_be.all():

            y[index_negative_be] = (y0[index_negative_be] * np.sqrt(np.abs(vals[index_negative_be])) + y0_diff[index_negative_be] + (s[index_negative_be] / np.sqrt(np.abs(vals[index_negative_be])))) / (
                        2 * np.sqrt(np.abs(vals[index_negative_be]))) * np.exp(np.sqrt(np.abs(vals[index_negative_be])) * t) + (
                                            y0[index_negative_be] * np.sqrt(np.abs(vals[index_negative_be])) - y0_diff[index_negative_be] + (s[index_negative_be] / np.sqrt(np.abs(vals[index_negative_be])))) / (
                                            2 * np.sqrt(np.abs(vals[index_negative_be]))) * np.exp(-np.sqrt(np.abs(vals[index_negative_be])) * t) - (s[index_negative_be]/np.abs(vals[index_negative_be]) )

            y_diff[index_negative_be] = (y0[index_negative_be] * np.sqrt(np.abs(vals[index_negative_be])) + y0_diff[index_negative_be]+ (s[index_negative_be] / np.sqrt(np.abs(vals[index_negative_be])))) / 2 * np.exp(
                np.sqrt(np.abs(vals[index_negative_be])) * t) - (y0[index_negative_be] * np.sqrt(np.abs(vals[index_negative_be])) - y0_diff[index_negative_be]+ (s[index_negative_be] / np.sqrt(np.abs(vals[index_negative_be])))) / 2 * np.exp(
                -np.sqrt(np.abs(vals[index_negative_be])) * t)


        if vals_negative_ae.all():
            y[index_negative_ae] = 1. / 2. * s_lambda_zero[index_negative_ae] * t * t + y0_diff[index_negative_ae] * t + y0[
                index_negative_ae]

            y_diff[index_negative_ae] = s_lambda_zero[index_negative_ae] * t + y0_diff[index_negative_ae]

        vecs_[z] = vecs
        y_[z] = y
        y_diff_[z] = y_diff

    return vecs_, y_, y_diff_

application_vv.name = 'applied_to_velocity_verlet' # add attribute to the function for marker
