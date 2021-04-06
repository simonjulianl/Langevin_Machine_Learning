import json

from MC_config import MC_config

if __name__=='__main__':

    data = { }

    data['nparticle'] = MC_config.nparticle
    data['temperature'] = MC_config.temperature
    data['mode'] = MC_config.mode
    data['init_path'] = MC_config.init_path
    data['filename'] = MC_config.filename
    data['seed'] = MC_config.seed
    data['DIM'] = MC_config.DIM
    data['rho'] = MC_config.rho
    data['nsamples'] = MC_config.nsamples
    data['mcstep'] = MC_config.mcstep
    data['max_energy'] = MC_config.max_energy
    data['DISCARD'] = MC_config.DISCARD
    data['interval'] = MC_config.interval
    data['dq'] = MC_config.dq

    with open('MC_config.tmpl', 'w') as outfile:
        json.dump(data, outfile,indent=4)


