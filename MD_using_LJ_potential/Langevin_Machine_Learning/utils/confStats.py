#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import warnings
import matplotlib.pyplot as plt
from ..phase_space import phase_space
import os


# from ..hamiltonian.pb import periodic_bc

class confStat:
    '''Helper Class to get the statistic of the configuration

    Configurations must always in pos/vel matrix of N X DIM dimensions

    '''

    @staticmethod
    def temp(**configuration):
        '''
        Helper Function to obtain the kinetic energy based on the momentum of particles

        Parameters
        ----------

        **kwargs :  configuration state consisting
            - vel : N X DIM matrix
                Velocity matrix of the configuration of N X DIM shape
            - N : int
                total number of particles
            - DIM : int
                Dimension of the particles
            - m : float
                mass of the particle
            - BoxSize : float
                scaling of the box cell
        Returns
        -------
        temperature : float
            The translational temperature of the configurations

        '''
        try:
            N = configuration['N']
            particle = configuration['particle']
            DIM = configuration['DIM']
            m = configuration['m']
            vel = configuration['phase_space'].get_p() / m  # v = p/m
            # pb = configuration['pb_q']
        except:
            raise Exception('N / Dimension / Mass / vel not supplied')

        try:
            BoxSize = configuration['BoxSize']
        except:
            BoxSize = 1
            warnings.warn('BoxSize not supplied, set to 1')

        # print('confStats.py temp vel',vel.shape)
        ene_kin = []
        for i in range(N):
            ene_kin_ = 0.0
            vel_ = vel[i, :]  # rescale each velocity according to the box size

            for j in range(particle):
                ene_kin_ += 0.5 * m * np.sum(np.multiply(vel_[j, :], vel_[j, :]), axis=0)  # 1/2 m v^2 for constant mass

            ene_kin.append(ene_kin_)

        ene_kin = np.array(ene_kin)

        ene_kin_aver = 1.0 * ene_kin / particle
        temperature = 2.0 / DIM * ene_kin_aver  # by equipartition theorem

        return temperature

    @staticmethod
    def kinetic_energy(**configuration):
        '''
        Helper Function to obtain the translational kinetic energy

        Parameters
        ----------

        **kwargs :  configuration state consisting
            - N : int
                total number of particles
            - DIM : int
                Dimension of the particles
        Returns
        -------
        ene_kin_aver : float
            The average translational kinetic energy of the configuration

        '''
        try:
            N = configuration['N']
            particle = configuration['particle']
            DIM = configuration['DIM']
            m = configuration['m']
            time_step = configuration['time_step']
            vel = configuration['phase_space'].get_p() / m  # v = p/m
        except:
            raise Exception('particle / DIM ot supplied')

        # ene_kin_aver = confStat.temp(**configuration) * DIM / 2
        ene_kin = []
        for i in range(N):
            ene_kin_ = 0.0
            vel_ = vel[i, :]  # rescale each velocity according to the box size

            for j in range(particle):
                ene_kin_ += 0.5 * m * np.sum(np.multiply(vel_[j, :], vel_[j, :]), axis=0)  # 1/2 m v^2 for constant mass

            ene_kin.append(ene_kin_)

        ene_kin = np.array(ene_kin)

        return ene_kin

    @staticmethod
    def plot_stat(q_hist: list, p_hist: list, mode: str, **configuration):

        line_plot = ['energy', 'p', 'potential', 'kinetic', 'q', 'all', 'instantaneous_temp','energy_dist']
        hist_plot = ['v_dist', 'q_dist', 'speed_dist']

        if mode not in line_plot and mode not in hist_plot:
            raise Exception('Modes not available , check the mode')

        assert q_hist.shape == p_hist.shape  # q list and p list must have the same size

        color = {
            'p': 'blue',
            'q': 'blue',
            'potential': 'orange',
            'kinetic': 'orange',
            'energy': 'orange',
            'q_dist': 'black',
            'v_dist': 'black',
            'speed_dist': 'black'
        }

        dim = {0: 'x', 1: 'y', 2: 'z'}
        hamiltonian = configuration['hamiltonian']
        time_step = configuration['tau']
        iterations = configuration['iterations']
        Temperature = configuration['Temperature']
        particle = configuration['particle']
        DIM = configuration['DIM']
        gamma = configuration['gamma']

        if mode in line_plot:
            potential = []
            kinetic = []

            for i in range(len(q_hist)):
                p_dummy_list = np.zeros(q_hist[i].shape)
                temporary_phase_space = phase_space()
                temporary_phase_space.set_q(q_hist[i])
                temporary_phase_space.set_p(p_dummy_list)

                potential.append(
                    hamiltonian.total_energy(temporary_phase_space, configuration['pb_q']))  # ADD periodicity=True
                configuration['phase_space'].set_p(p_hist[i])

                kinetic_ = confStat.kinetic_energy(**configuration)
                kinetic.append(kinetic_)

            kinetic = np.array(kinetic).transpose()
            potential = np.array(potential).transpose()
            energy = kinetic + potential

            if mode == 'p' or mode == 'q':  # for p and q we plot dimension per dimension
                for n in range(configuration['DIM']):
                    if mode == 'p':
                        plt.plot(p_hist[:, :, :, n], color=color[mode], label='p')
                    elif mode == 'q':
                        plt.plot(q_hist[:, :, :, n], color=color[mode], label='q')
                    plt.xlabel('sampled steps')
                    plt.ylabel(mode + ' ' + dim[n])
                    plt.legend(loc='best')
                    plt.show()

            elif mode == 'all':

                filename_creater = os.path.abspath('Method_vv/N{}/T{}/ts{}'.format(particle, Temperature, time_step))
                if not os.path.exists(filename_creater):
                    os.makedirs(filename_creater)

                for i in range(energy.shape[0]):
                    if i < 1:
                        t = np.arange(0., iterations * time_step + time_step, time_step)
                        plt.plot(t, energy[i], label='total energy')
                        plt.plot(t, kinetic[i], label='kinetic energy')
                        plt.plot(t, potential[i], label='potential energy')

                        plt.title('sample {}; Temp {}; gamma {}; iterations {}; time step {}'.format(i, Temperature, gamma,iterations,
                                                                                           time_step))
                        plt.xlabel('time_step')
                        plt.legend(loc='best')
                        # plt.show()
                        plt.savefig(filename_creater + '/s{}_T{}_ts{}_gamma{}_E.png'.format(i, Temperature, time_step,gamma))
                        plt.close()


            elif mode == 'instantaneous_temp':
                T_t = []

                for i in range(p_hist.shape[0]):  # iteration
                    m = 1
                    ene_kin_ = 0.0
                    vel_ = p_hist[i, 0, :]
                    for j in range(p_hist.shape[2]):  # N_particle
                        ene_kin_ += 0.5 * m * np.sum(np.multiply(vel_[j, :], vel_[j, :]), axis=0)

                    ene_kin_aver = 1.0 * ene_kin_ / particle
                    temperature = 2.0 / DIM * ene_kin_aver
                    T_t.append(temperature)

                avg_temp = np.sum(np.array(T_t))/p_hist.shape[0]
                t = np.arange(0., iterations * time_step + time_step, time_step)
                plt.plot(t, T_t, color='red', label=mode)
                plt.axhline(avg_temp, color='black',label='avg_temp={:.3f}'.format(avg_temp))
                plt.xlabel('time_step')
                plt.ylabel(mode)
                plt.legend(loc='best')
                plt.show()

            elif mode == 'energy_dist':
                # plot exact
                print(energy.shape)
                e = np.linspace(np.min(energy), np.max(energy), 1000)
                prob_e = ((_m * _beta) / (2 * np.pi)) ** 0.5 * np.exp(-1 / (configuration['kB'] * configuration['Temperature']) * e)

                plt.plot(v, prob_v, marker=None, color="red", linestyle='-', label='v exact')


            else:

                for i in range(energy.shape[0]):

                    if i < 4:
                        t = np.arange(0., iterations * time_step + time_step, time_step)
                        if mode == 'energy':  # if energy , we use average on every dimension
                            plt.plot(t, energy[i], color=color[mode], label='total energy')
                        elif mode == 'kinetic':
                            plt.plot(t, kinetic[i], color=color[mode], label='kinetic energy')
                        elif mode == 'potential':
                            plt.plot(t, potential[i], color=color[mode], label='potential energy')

                        plt.xlabel('time_step')
                        plt.ylabel(mode)
                        plt.legend(loc='best')
                        plt.show()

        else:
            try:
                _beta = 1 / (configuration['kB'] * configuration['Temperature'])
                _m = configuration['m']
            except:
                raise Exception('kB / Temperature not set ')

            for n in range(configuration['DIM']):
                if mode == 'q_dist':
                    print('confState.py q_hist', q_hist.shape)
                    print(q_hist[:, :, :, n].shape)
                    curr = q_hist[:, :, :, n].reshape(-1, 2)  # collapse to 1 long list
                    print(curr.shape)

                    # plot exact
                    q = np.linspace(np.min(curr), np.max(curr), 1000)
                    # create hamiltonian list
                    potential = np.array([])
                    for i in range(len(q)):
                        q_list_temp = np.expand_dims(q[i], axis=0).reshape(2, 2)
                        print('confState.py q_list_temp', q_list_temp)

                        p_list_temp = np.zeros(q_list_temp.shape)  # prevent KE from integrated
                        potential = np.append(potential,
                                              hamiltonian.total_energy(q_list_temp, p_list_temp))
                    prob_q = np.exp(-_beta * potential)
                    dq = q[1:] - q[:-1]
                    yqs = 0.5 * (prob_q[1:] + prob_q[:-1])
                    Zq = np.dot(yqs.T, dq)  # total area
                    plt.plot(q, prob_q / Zq, marker=None, color="red", linestyle='-', label='q exact')

                elif mode == 'v_dist':
                    # print('p_hist',p_hist_.shape)
                    curr = p_hist[:, 0, :, n].reshape(-1, 1) / _m  # collapse to 1 long list  sample 0
                    # print('curr', curr.shape)
                    # print(np.min(curr))
                    # print(np.max(curr))
                    # plot exact
                    v = np.linspace(np.min(curr), np.max(curr), 1000)
                    prob_v = ((_m * _beta) / (2 * np.pi)) ** 0.5 * np.exp(-_beta * (v ** 2.0) / 2)

                    plt.plot(v, prob_v, marker=None, color="red", linestyle='-', label='v exact')

                elif mode == 'speed_dist':
                    curr = p_hist[:, :, :].reshape(-1, configuration['DIM']) / _m  # collapse to 1 long list of N X DIM
                    speed = np.linalg.norm(curr, 2, axis=1)
                    v = np.linspace(np.min(speed), np.max(speed), 1000)
                    prob_v = 4 * np.pi * (_beta / (2 * np.pi)) ** 1.5 * v ** 2.0 * np.exp(-_beta * v ** 2.0 / (2 * _m))
                    plt.plot(v, prob_v, marker=None, color="red", linestyle='-', label='v exact')

                interval = (np.max(curr) - np.min(curr)) / 50
                values, edges = np.histogram(curr, bins=np.arange(np.min(curr), np.max(curr), interval),
                                             density=True)  # plot pdf
                center_bins = 0.5 * (edges[1:] + edges[:-1])
                plt.plot(center_bins, values, color=color[mode], label=mode)
                plt.ylabel('pdf')
                plt.legend(loc='best')

                if mode == 'speed_dist':
                    plt.xlabel(mode[0])
                    plt.show()
                    return  # exit the function since we take all the dimensions at once

                plt.xlabel(mode[0] + dim[n])
                plt.show()
