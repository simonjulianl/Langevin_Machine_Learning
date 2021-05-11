from hamiltonian    import hamiltonian
from lennard_jones  import lennard_jones
from kinetic_energy import kinetic_energy


class noML_hamiltonian(hamiltonian):

    ''' this is a hamiltonian with no ML '''

    _obj_count = 0

    def __init__(self):

        noML_hamiltonian._obj_count += 1
        assert (noML_hamiltonian._obj_count == 1), type(self).__name__ + " has more than one object"

        super().__init__()

        super().append(lennard_jones())
        super().append(kinetic_energy())
        print('noML_hamiltonian initialized')


    def set_tau(self, tau_cur):
        return

