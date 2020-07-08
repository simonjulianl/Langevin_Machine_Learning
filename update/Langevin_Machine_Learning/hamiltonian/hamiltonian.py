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
        
    def total_energy(self, q_list, p_list, BoxSize = 1,periodicity = False):
        '''
        get the hamiltonian which is define as H(p,q) for every separable terms

        Returns
        -------
        H : float
            H is the hamiltonian of the states with separable terms

        '''
        
        H = 0 # hamiltonian container
        for term in self.hamiltonian_terms :
            print('hamiltonian.py term',term)
            print('hamiltonian.py q_list',q_list)
            print('hamiltonian.py p_list',p_list)
            if 'Lennard_Jones' in term.__class__.__name__:
                H += term.energy(q_list, p_list, BoxSize,periodicity)
            else:
                H += term.energy(q_list, p_list,periodicity)

            print('hamiltonian.py periodicity term', periodicity, term)
    
        return H

    def dHdq(self, q_list, p_list, BoxSize = 1, periodicity = False):
        '''
        Function to get dHdq for every separable terms 

        Returns
        -------
        dHdq : float 
            dHdq is the derivative of H with respect to q for N X DIM dimension 
        '''
        assert q_list.shape == p_list.shape and len(q_list.shape) == 3
        dHdq = np.zeros(q_list.shape)
        print('Hamiltonian.py dHdq', dHdq.shape)
        print('Hamiltonian.py hamiltonian_terms', self.hamiltonian_terms)

        for term in self.hamiltonian_terms :
            if 'Lennard_Jones' in term.__class__.__name__ : # Exception because need to scale by boxsize
                print('Hamiltonian.py for dHdq', dHdq)
                dHdq += term.evaluate_derivative_q(q_list, p_list, BoxSize, periodicity)
                print('Hamiltonian.py dHdq+', dHdq.shape)
            else:
                print('Hamiltonian.py for dHdq', dHdq)
                dHdq += term.evaluate_derivative_q(q_list, p_list, periodicity)
                print('Hamiltonian.py dHdq+', dHdq.shape)

        return dHdq 
    
    def dHdp(self, q_list, p_list, periodicity = False):
        '''
        Function to get dHdp for every separable terms 

        Returns
        -------
        dHdp : float 
            dHdqp is the derivative of H with respect to p for N X DIM dimension 
        '''
        assert q_list.shape == p_list.shape and len(q_list.shape) == 3
        dHdp = np.zeros(q_list.shape)
        print('Hamiltonian.py dHdp',dHdp.shape)
        for term in self.hamiltonian_terms : 
            dHdp += term.evaluate_derivative_p(q_list, p_list, periodicity)
            
        return dHdp
        
    def __repr__(self):
        ''' return list of terms'''
        terms = ''
        for term in self.hamiltonian_terms :
            terms += term._name + '\n'
        return terms 