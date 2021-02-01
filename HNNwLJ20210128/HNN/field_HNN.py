from HNNwLJ20210128.parameters.MD_paramaters import MD_parameters
from HNNwLJ20210128.parameters.ML_paramaters import ML_parameters
from HNNwLJ20210128.hamiltonian.hamiltonian import hamiltonian
from HNNwLJ20210128.hamiltonian.lennard_jones import lennard_jones
from HNNwLJ20210128.hamiltonian.kinetic_energy import kinetic_energy

class field_HNN(hamiltonian):

    def __init__(self, network):

        super().__init__()
        self.network = network

        # # append term to calculate dHdq
        super().append(lennard_jones())
        super().append(kinetic_energy(MD_parameters.mass))

    def train(self):
        self.network.train()  # pytorch network for training

    def eval(self):
        self.network.eval()


    def dHdq(self, phase_space):


        q_list = phase_space.get_q()
        p_list = phase_space.get_p()

        noML_dHdq = super().dHdq(phase_space)

        phase_space.set_q(q_list)
        phase_space.set_p(p_list)

        data = self.phase_space2data(phase_space, MD_parameters.tau_long)
        predict = self.network(data, MD_parameters.nparticle, MD_parameters.DIM)
        corrected_dHdq = noML_dHdq.to(ML_parameters.device) + predict  # in linear_vv code, it calculates grad potential.

        return corrected_dHdq

    def phase_space2data(self, phase_space, tau_cur):

        q_list = phase_space.get_q()
        p_list = phase_space.get_p()