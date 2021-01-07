
class pairwise_HNN:

    def __init__(self,hamiltonian,network):
        self.network = network
        self.noML_hamiltonian = hamiltonian

    def train(self):
        self.network.train() # pytorch network for training


    def eval(self):
        self.network.eval()

    def dHdq(self, phase_space, pb):
        data = self.phase_space2data(phase_space,pb)
        predict = self.network(data)
        noML_force = self.noML_hamiltonian.dHdq(phase_space,pb)

        corrected_force = noML_force + predict

        return corrected_force

     def phase_space2data(self,phase_space,pb):

         return data