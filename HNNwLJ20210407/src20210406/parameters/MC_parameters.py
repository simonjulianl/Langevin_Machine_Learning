import json
import math

class MC_parameters:

    # open when run MC
    nparticle = None
    rho = None           # density
    boxsize = None

    temperature = None
    mode = None                               # set mode train or valid or test for filename

    init_path = None                          # path to write data
    filename  = None                          # filename to write data

    seed = None                               # set different seed for generate data for train/valid/test
                                              # nsamples 20900 ->  23645 for train / 35029 for valid
                                              # nsamples 41800 ->  89236 for train / 49832 for valid
                                              # nsamples 1000  -> 15343 for test

    DIM = None
    nsamples = None                           # the number of samples for mc
    mcstep = None                             # mc step each sample
    max_energy = None                         # energy threshold
    DISCARD = None                            # discard initial mc steps
    interval = None                           # take mc step every given interval
    dq = None                                 # displacement to increase acceptance rate

    iterations = None

    # 4 particles : T0.04 -> 0.015, T0.16 -> 0.035, T0.32 -> 0.08  DISCARD 4000  interval 40
    # 8 particles :       -> 0.01         -> 0.023,       -> 0.04  DISCARD 8000  interval 40
    # 16 particles :      -> 0.006        -> 0.015        -> 0.02  DISCARD 16000 interval 40

    @staticmethod
    def load_dict(json_filename):
        with open(json_filename) as f:
            data = json.load(f)

        MC_parameters.nparticle = data['nparticle']
        MC_parameters.temperature = data['temperature']
        MC_parameters.mode = data['mode']                       # set mode train or valid or test for filename
    
        MC_parameters.init_path = data['init_path']
        MC_parameters.filename  = data['filename']
    
        MC_parameters.seed = data['seed']                       # set different seed for generate data for train/valid/test
                                                                # nsamples 20900 ->  23645 for train / 35029 for valid
                                                                # nsamples 41800 ->  89236 for train / 49832 for valid
                                                                # nsamples 1000  -> 15343 for test
        MC_parameters.DIM = data['DIM']
        MC_parameters.rho = data['rho']                         # density
    
        MC_parameters.nsamples = data['nsamples']               # the number of samples for mc
        MC_parameters.mcstep = data['mcstep']                   # mc step each sample
        MC_parameters.max_energy = data['max_energy']           # energy threshold
        MC_parameters.DISCARD = data['DISCARD']                 # discard initial mc steps
        MC_parameters.interval = data['interval']               # take mc step every given interval
        MC_parameters.dq = data['dq']                           # displacement to increase acceptance rate

        MC_parameters.boxsize = math.sqrt(MC_parameters.nparticle/MC_parameters.rho)
        MC_parameters.iterations = MC_parameters.mcstep*MC_parameters.interval+MC_parameters.DISCARD




    
