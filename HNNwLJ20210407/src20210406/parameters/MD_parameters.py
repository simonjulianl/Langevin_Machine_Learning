import json

from integrator.methods import linear_velocity_verlet
from hamiltonian.noML_hamiltonian import noML_hamiltonian
from HNN.pairwise_HNN import pairwise_HNN
from HNN.models.pairwise_MLP import pairwise_MLP

class MD_parameters:

    init_qp_path       = None             # path to read data
    data_path 	       = None             # path to write data
    init_qp_filename   = None             # filename to read data
    data_filenames     = None             # filename to write data

    # open when run MD
    tau_short          = None             # short time step for label
    tau_long           = None             # value of tau_long

    append_strike      = None             # number of short steps to make one long step
    niter_tau_long     = None             # number of MD steps for long tau
    save2file_strike   = None             # number of short steps to save to file
    niter_tau_short    = None             # number of MD steps for short tau

    integrator_method = linear_velocity_verlet.linear_velocity_verlet
    integrator_method_backward = linear_velocity_verlet.linear_velocity_verlet_backward

    # hamiltonian_obj = noML_hamiltonian()
    hamiltonian_obj = pairwise_HNN(pairwise_MLP(), pairwise_MLP())

    @staticmethod
    def load_dict(json_filename):
        with open(json_filename) as f:
            data = json.load(f)

        MD_parameters.init_qp_path      = data['init_qp_path']
        MD_parameters.data_path         = data['data_path']
        MD_parameters.init_qp_filename  = data['init_qp_filename']
        MD_parameters.data_filenames    = data['data_filenames']

        MD_parameters.tau_short         = data['tau_short']
        MD_parameters.append_strike     = data['append_strike']
        MD_parameters.niter_tau_long    = data['niter_tau_long']
        MD_parameters.save2file_strike  = data['save2file_strike']
        MD_parameters.tau_long          = data['tau_long']
        MD_parameters.niter_tau_short   = data['niter_tau_short']

        MD_parameters.hamiltonian_obj.set_tau(MD_parameters.tau_long)
