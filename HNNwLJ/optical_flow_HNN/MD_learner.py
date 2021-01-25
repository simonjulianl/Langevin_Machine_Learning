import matplotlib.pyplot as plt
import torch
import shutil
import time

class MD_learner:

    def __init__(self,linear_integrator, noML_hamiltonian, optical_flow_HNN, **state):

        self.linear_integrator = linear_integrator
        self.noML_hamiltonian =noML_hamiltonian
        self.optical_flow_HNN = optical_flow_HNN
        self._state = state

        self._MLP = state['MLP'].to(state['_device'])
        self._opt = state['opt']
        self._loss = state['loss']

        self._current_epoch = 1
        # initialize best models
        self._best_validation_loss = float('inf')


    def phase_space2label(self, MD_integrator, noML_hamiltonian):
        label = MD_integrator.integrate(noML_hamiltonian)
        return label

    # phase_space consist of minibatch data
    # def trainer(self, filename):





