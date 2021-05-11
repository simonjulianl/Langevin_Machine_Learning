import json

from integrator.methods import linear_velocity_verlet

class MD_parameters:

    MC_init_config_filename      = None             # filename to read data
    MD_data_dir                  = None             # path to write data
    MD_output_basenames          = None             # filename to write data

    tau_short                    = None             # short time step for label
    tau_long                     = None             # value of tau_long

    append_strike                = None             # number of short steps to make one long step
    niter_tau_long               = None             # number of MD steps for long tau
    save2file_strike             = None             # number of short steps to save to file
    niter_tau_short              = None             # number of MD steps for short tau

    hamiltonian_type             = None

    integrator_method = linear_velocity_verlet.linear_velocity_verlet
    integrator_method_backward = linear_velocity_verlet.linear_velocity_verlet_backward

    @staticmethod
    def load_dict(json_filename):
        with open(json_filename) as f:
            data = json.load(f)

        MD_parameters.MC_init_config_filename     = data['MC_init_config_filename']
        MD_parameters.MD_data_dir                 = data['MD_data_dir']
        MD_parameters.MD_output_basenames         = data['MD_output_basenames']

        MD_parameters.tau_short                   = data['tau_short']
        MD_parameters.append_strike               = data['append_strike']
        MD_parameters.niter_tau_long              = data['niter_tau_long']
        MD_parameters.save2file_strike            = data['save2file_strike']
        MD_parameters.tau_long                    = data['tau_long']
        MD_parameters.niter_tau_short             = data['niter_tau_short']

        MD_parameters.hamiltonian_type            = data['hamiltonian_type']
