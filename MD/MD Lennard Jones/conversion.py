#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 11:21:29 2020

@author: simon
"""
#this is Argon gas data
sigma = 3.4e-10
mass = 6.634e-26
kB = 1.381e-23
epsilon = 1.653e-21

def convertSI_length(length):
    return length * sigma

def convertSI_energy(energy):
    return energy * epsilon

def convertSI_mass(m):
    return m * mass

def convertSI_velocity(vel):
    return vel * (epsilon/mass) ** 0.5

def convertSI_temp(temp):
    return temp * epsilon / kB

def convertSI_force(force):
    return force * epsilon / sigma

def convertSI_time(time):
    return time * sigma * (mass/epsilon) ** 0.5

def convertRU_length(length):
    return length / sigma
    
def convertRU_energy(energy):
    return energy/epsilon

def convertRU_mass(m):
    return m / mass

def convertRU_velocity(vel):
    return vel / (epsilon/mass) ** 0.5

def convertRU_temp(temp):
    return temp / epsilon * kB

def convertRU_force(force):
    return force / epsilon * sigma

def convertRU_time(time):
    return time / sigma / (mass/epsilon) ** 0.5


