import os
import sys
import json

class MD_config:

    data_dir    = '../data/gen_by_MD/'

    tau_short = 0.001                                   # short time step for label
    append_strike = 100                                 # number of short steps to make one long step
    niter_tau_long  = 1                                 # number of MD steps for long tau
    save2file_strike = 100                              # number of short steps to save to file; save2file_strike >= append_strike
    tau_long = append_strike * tau_short                # value of tau_long
    niter_tau_short = niter_tau_long * append_strike    # number of MD steps for short tau


if __name__=='__main__':

    argv = sys.argv

    if len(argv) != 5:
        print('usage <programe> <MC_init_config_basename> <basename> <hamiltonian_type : noML or pair_wise_HNN> <json_dir>')
        quit()

    MC_init_config_basename   = argv[1]              # init_dir + basename (no extension e.g. .pt)
    basename                  = argv[2]
    hamiltonian_type          = argv[3]
    json_dir                  = argv[4]

    MC_init_config_filename = MC_init_config_basename + '.pt'    # copy init file
    MD_data_dir             = MD_config.data_dir + basename
    MD_output_basenames     = MD_data_dir + '/' + basename

    data = { }

    data['MC_init_config_filename'] = MC_init_config_filename
    data['MD_data_dir'] = MD_data_dir
    data['MD_output_basenames'] =MD_output_basenames
    data['tau_short'] = MD_config.tau_short
    data['append_strike'] = MD_config.append_strike
    data['niter_tau_long'] = MD_config.niter_tau_long
    data['save2file_strike'] = MD_config.save2file_strike
    data['tau_long'] = MD_config.tau_long
    data['niter_tau_short'] = MD_config.niter_tau_short
    data['hamiltonian_type'] = hamiltonian_type

    json_dir_name = json_dir + basename

    if not os.path.exists(json_dir_name):
        os.makedirs(json_dir_name)

    json_file_name = json_dir_name + '/MD_config.dict'
    with open(json_file_name, 'w') as outfile:
        json.dump(data, outfile,indent=4)


