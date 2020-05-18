#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 13:50:17 2020

@author: simon
"""


def pot_r_ewald(pos,q,kappa,BoxSize = 10.0):
    import numpy as np
    from itertools import combinations
    from scipy.special import erfc  # erfc = 1 - erf
    #where erf is just the integration of gaussian function
    #in this model, we use smeared charges of gaussian function
    
    n,d = pos.shape
#    assert d == 3, 'r dimension error'
#    assert n == q.size,'q dimension error'
#    
    pot = 0.0
    
    for i,j in combinations(range(n),2):
        Sij = pos[i,:] - pos[j,:]
        #periodic boundary condition
        for k in range(d):
            period = np.where(pos[:,k]> 0.5)
            pos[period,j] = pos[period,j] -1.0
            period = np.where(pos[:,k] < -0.5)
            pos[period,j] = pos[period,j] + 1.0
        
        Rij = Sij  * BoxSize
        Rsqij = np.dot(Rij,Rij)
        
        MagR = np.sqrt(Rsqij)
        Vij = q[i] * q[j] * erfc (kappa * MagR) / MagR
        pot += Vij
        
    return pot

def pot_k_ewald(pos,nk,q,kappa,BoxSize):
    import numpy as np
    from itertools import product # cartesian product
    global first_call,k_sq_max,kfac
    
    n,d = pos.shape
    
    #assert if necessary
    twopi= 2.0 * np.pi
    twopi_sq = twopi ** 2
    
    if first_call:
        b = 1.0 / 4.0 / kappa ** 2
        k_sq_max = nk ** 2
        kfac = np.zeros(k_sq_max + 1,dtype = np.float_)
        
        for kx,ky,kz in product(range(nk+1),repeat = 3):
            k_sq = kx ** 2 + ky ** 2 + kz ** 2
            if k_sq <= k_sq_max and k_sq > 0:
                kr_sq = twopi_sq * k_sq
                kfac[k_sq] = twopi * np.exp(-b * kr_sq) / kr_sq
        first_call =False
        
        
    eikx = np.zeros((n,nk + 1), dtype = np.complex_)
    eiky = np.zeros((n,2 * nk + 1),dtype = np.complex_)
    eikz = np.zeros((n,2 * nk + 1), dtype = np.complex_)
    
    eikx[:, 0] = 1.0 + 0.0j
    eiky[:, nk + 0] = 1.0 + 0.0j
    eikz[:, nk + 0] = 1.0 + 0.0j
    
    eikx[: , 1] = np.cos(twopi * pos[:,0] * BoxSize ) + np.sin(twopi * pos[:,0] * BoxSize) * 1j
    eiky[: , nk+1] = np.cos(twopi * pos[:,1]* BoxSize ) + np.sin(twopi * pos[:,1]* BoxSize) * 1j
    eikz[:,nk+1] = np.cos(twopi * pos[:,2]* BoxSize ) + np.sin(twopi * pos[:,2]* BoxSize) * 1j
    
    for k in range(2,nk+1):
        eikx[:,k] = eikx[:,k-1] * eikx[:,1]
        eiky[:,nk+k] = eiky[:,nk+k -1] * eiky[:,nk+1]
        eikz[:,nk+k] = eikz[:,nk+k - 1] * eikz[:,nk+1]
        
    eiky[:,0:nk] = np.conj(eiky[:,2*nk : nk: -1])
    eikz[:,0:nk] = np.conj(eikz[:,2*nk : nk : -1])
    
    pot = 0.0
    
    for kx in range(nk + 1):
        factor = 1.0 if kx == 0 else 2.0 # apparently account for skipping negative, dont really understand
        
        for ky,kz in product(range(-nk,nk+1) , repeat = 2):
            k_sq = kx ** 2 +  ky ** 2 + kz ** 2
            
            if k_sq <= k_sq_max and k_sq > 0:
                term = np.sum(q[:] * eikx[:,kx] * eiky[:,nk+ky] * eikz[:,nk+kz])
                pot += factor * kfac[k_sq] * np.real(np.conj(term) * term)
                
    pot -= kappa * np.sum ( q ** 2) / np.sqrt(np.pi)  #self term 
    return pot

 