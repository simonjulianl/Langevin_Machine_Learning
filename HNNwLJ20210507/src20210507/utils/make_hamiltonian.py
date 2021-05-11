from hamiltonian.noML_hamiltonian   import noML_hamiltonian
from HNN.pairwise_HNN               import pairwise_HNN
from HNN.fields_HNN                 import fields_HNN
from HNN.models.pairwise_MLP        import pairwise_MLP
from HNN.models.fields_cnn          import fields_cnn
from utils.interpolator             import interpolator

def make_hamiltonian(hamiltonian_type, integrator, tau_short, tau_long, ML_param):

    if hamiltonian_type  == 'noML':
        hamiltonian_obj = noML_hamiltonian()

    elif hamiltonian_type == 'pairwise_HNN':
        net1 = pairwise_MLP(ML_param.MLP_input, ML_param.MLP_nhidden)
        net2 = pairwise_MLP(ML_param.MLP_input, ML_param.MLP_nhidden)
        hamiltonian_obj = pairwise_HNN(net1, net2)
        hamiltonian_obj.set_tau(tau_long)

    elif hamiltonian_type == 'fields_HNN':
        net = fields_cnn(ML_param.gridL, ML_param.batch_size, ML_param.cnn_channels) # HK
        interpolator_obj = interpolator() # SJ
        hamiltonian_obj = fields_HNN(net,integrator, interpolator_obj)
        hamiltonian_obj.set_tau_short(tau_short)
        hamiltonian_obj.set_tau_long(tau_long)  # HK checked not call this function anywhere when this hamiltonian type

    else:
        assert (False), 'invalid hamiltonian type given'


    return hamiltonian_obj
