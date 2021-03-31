import torch
import glob
import re

def keyFunc(afilename):
    '''function to change the sort from 'ASCIIBetical' to numeric by isolating the number
        in the filename'''

    nondigits = re.compile("\D")
    return int(nondigits.sub("", afilename))

def qp_list_combine(filename):
    ''' function to combine files

    filename : string
            all files to combine
    qp_list : torch.tensor
            shape is [ (q,p), nsamples, nparticle, DIM ] combined along different temp

    return : list
            add qp_list to the list
    '''

    qp_list_app = []

    for i in range(len(filename)):

        print(filename[i])
        qp_list = torch.load(filename[i])
        print(qp_list.shape)
        qp_list_app.append(qp_list)

    qp_list_app = torch.cat(qp_list_app, dim=1)

    return qp_list_app

if __name__ == '__main__':

    ''' combine initial q, p at different temperature for training  '''
    # paramters
    nparticle = 2
    mode      = 'train'  # set train or valid

    # io varaiables
    root_path = '../init_config/n{}/'.format(nparticle)
    filename = sorted(glob.glob(root_path + 'run*/' + 'nparticle{}_new_nsim_rho0.1_T'.format(nparticle) + '*' + '_pos_{}_sampled.pt'.format(mode)), key=keyFunc)
    write_filename = root_path + 'nparticle{}_new_nsim_rho0.1_allT_pos_{}_sampled.pt'.format(nparticle, mode)

    qp_list_app = qp_list_combine(filename)
    torch.save(qp_list_app, write_filename)
