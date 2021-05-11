import json
import math
import sys
import os

class MC_config:

    init_dir           = '../data/gen_by_MC/'

    DIM = 2
    rho = 0.1                            # density

    mcstep = 100                        # mc step each sample  default : 100
    max_energy = 1e3                    # energy threshold
    DISCARD = 32000                     # discard initial mc steps default : 32000
    interval = 40                       # take mc step every given interval  default : 40


if __name__=='__main__':

    argv = sys.argv

    if len(argv) != 6:
        print('usage <programe> <nparticle> <temperature> <seed> <nsamples> <dq>')
        quit()

    nparticle   = int(argv[1])
    temperature = float(argv[2])
    seed        = int(argv[3])
    nsamples    = int(argv[4])
    dq          = float(argv[5])

    basename = 'n' + str(nparticle) + 'T' + str(temperature) + 'seed' + str(seed) + 'nsamples' + \
               str(nsamples)

    dir_name = MC_config.init_dir + basename         # folder name
    MC_output_filename = dir_name + '/' + basename + '.pt'     # pathname

    data = { }

    data['nparticle'] = nparticle
    data['temperature'] = temperature
    data['MC_output_filename'] = MC_output_filename
    data['seed'] = seed
    data['DIM'] = MC_config.DIM
    data['rho'] = MC_config.rho
    data['nsamples'] = nsamples
    data['mcstep'] = MC_config.mcstep
    data['max_energy'] = MC_config.max_energy
    data['DISCARD'] = MC_config.DISCARD
    data['interval'] = MC_config.interval
    data['dq'] = dq

    # make directory

    json_dir_name = '../gen_by_MC/' + basename

    if not os.path.exists(json_dir_name):  # HK : give this if want change parameters in exited file
                os.makedirs(json_dir_name)
    #os.mkdir(json_dir_name)

    json_file_name = json_dir_name + '/MC_config.dict'
    with open(json_file_name, 'w') as outfile:
        json.dump(data, outfile,indent=4)


