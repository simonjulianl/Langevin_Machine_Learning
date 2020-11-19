#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 12:10:44 2020

@author: simon
"""


import numpy as np
from .Interaction import Interaction

class Hamiltonian:
    '''Class container for Hamiltonian
    
    Common Parameters 
    -----------------
    q_list : np.array
            q_list np.array of N X DIM matrix which is the position of the states
    p_list : np.array
        p_list np.array of N X DIM matrix which is the momentum of the states 
    '''
    
    def __init__(self):
        '''
        Hamiltonian class for all potential and kinetic interactions 


        '''
        self.hamiltonian_terms = [] # for every separable terms possible
        
    def append(self, term):
        '''
        helper function to add into the hamiltonian terms 

        Parameters
        ----------
        term : Interaction
            the interaction term as defined per each hamiltonian term of class Interaction

        '''
        if not isinstance(term, Interaction) :
            raise Exception('Interaction term is not derived from Interaction class')

        self.hamiltonian_terms.append(term)
        
    def total_energy(self, phase_space,pb):
        '''
        get the hamiltonian which is define as H(p,q) for every separable terms

        Returns
        -------
        H : float
            H is the hamiltonian of the states with separable terms

        '''
        H = 0
        #H = 0 # hamiltonian container
        for term in self.hamiltonian_terms :
            #print('hamiltonian.py term',term)
            #print('hamiltonian.py phase_space',phase_space)
            H += term.energy(phase_space,pb)
            #print('hamiltonian.py H',H)
            #print('hamiltonian.py periodicity term', periodicity, term)
    
        return H

    def dHdq(self,phase_space, pb):
        '''
        Function to get dHdq for every separable terms 

        Returns
        -------
        dHdq : float 
            dHdq is the derivative of H with respect to q for N X DIM dimension 
        '''
        q_list = phase_space.get_q()
        dHdq = np.zeros(q_list.shape)

        #print('Hamiltonian.py dHdq', dHdq.shape)
        #print('Hamiltonian.py hamiltonian_terms', self.hamiltonian_terms)

        for term in self.hamiltonian_terms :
            #print('Hamiltonian.py for dHdq', dHdq)
            #print('Hamiltonian.py hamiltonian term ', term)
            dHdq += term.evaluate_derivative_q(phase_space, pb)
            #print('Hamiltonian.py dHdq+', dHdq)

        return dHdq 
    
    def d2Hdq2(self,phase_space, pb):
        '''
        Function to get dHdp for every separable terms 

        Returns
        -------
        dHdp : float 
            dHdqp is the derivative of H with respect to p for N X DIM dimension 
        '''
        q_list = phase_space.get_q()

        N, N_particle,DIM  = q_list.shape
        d2Hdq2 = np.zeros((N,2*N_particle,2*N_particle))
        print(d2Hdq2.shape)
        #print('Hamiltonian.py dHdp',dHdp.shape)
        for term in self.hamiltonian_terms :
            print('Hamiltonian.py hamiltonian term ', term)
            d2Hdq2 += term.evaluate_second_derivative_q(phase_space, pb)
            
        return d2Hdq2
        
    def __repr__(self):
        ''' return list of terms'''
        terms = ''
        for term in self.hamiltonian_terms :
            terms += term._name + '\n'
        return terms 