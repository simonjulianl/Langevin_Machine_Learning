import json

from MD_config import MD_config

if __name__=='__main__':

    data = { }

    data['init_qp_path'] = MD_config.init_qp_path
    data['data_path'] = MD_config.data_path
    data['init_qp_filename'] = MD_config.init_qp_filename
    data['data_filenames'] = MD_config.data_filenames
    data['tau_short'] = MD_config.tau_short
    data['append_strike'] = MD_config.append_strike
    data['niter_tau_long'] = MD_config.niter_tau_long
    data['save2file_strike'] = MD_config.save2file_strike
    data['tau_long'] = MD_config.tau_long
    data['niter_tau_short'] = MD_config.niter_tau_short

    with open('MD_config.tmpl', 'w') as outfile:
        json.dump(data, outfile,indent=4)


