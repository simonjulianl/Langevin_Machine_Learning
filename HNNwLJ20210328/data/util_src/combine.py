import sys
import os

sys.path.append(os.path.abspath("../../src20210328"))
sys.path.append(os.path.abspath("../../src20210328/parameters"))

import torch
from MC_parameters              import MC_parameters
from MD_parameters              import MD_parameters
from utils.data_io              import data_io

def qp_list_combine(init_filename, mode):
    ''' function to combine qp list at different temp

    init_filename : string
    qp_list : torch.tensor
            shape is [ (q,p), nsamples, nparticle, DIM ]
    mode : str

    return : list
            add qp_list to the list
    '''

    qp_list_app = []
    for i, temp in enumerate(MD_parameters.temp_list):

        # qp_list shape is [(q,p), nsamples, nparticle, DIM]
        qp_list = data_io_obj.read_init_qp(init_filename + 'T{}_pos_'.format(temp) + str(mode) + '_sampled.pt')
        qp_list_app.append(qp_list)

    qp_list_app = torch.cat(qp_list_app, dim=1)  # concat along nsampless

    return qp_list_app

if __name__ == '__main__':

    uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])
    base_dir = uppath(__file__, 2)

    root_path = base_dir + '/init_config/'
    data_io_obj = data_io(root_path)

    nparticle = MC_parameters.nparticle
    rho = MC_parameters.rho
    mode = MC_parameters.mode


    init_filename = 'nparticle{}_new_nsim_rho{}_'.format(nparticle, rho)
    filename = 'nparticle{}_new_nsim_rho{}_T_all_pos_{}_sampled.pt'.format(nparticle, rho, mode)

    qp_list_app = qp_list_combine(init_filename, mode)

    data_io_obj.write_init_qp(filename, qp_list_app)