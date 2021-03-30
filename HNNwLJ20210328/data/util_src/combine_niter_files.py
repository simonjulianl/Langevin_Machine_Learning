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

    init_filename : string
    qp_list : torch.tensor
            shape is [niter, (q,p), nsamples, nparticle, DIM]  combined along niter

    return : list
            add qp_list to the list
    '''

    qp_list_app = []

    for i in range(len(filename)):

        print(filename[i])
        qp_list = torch.load(filename[i])
        print(qp_list.shape)
        qp_list_app.append(qp_list)

    qp_list_app = torch.cat(qp_list_app, dim=0)

    return qp_list_app

if __name__ == '__main__':

    # paramters
    nparticle = 2
    temp      = 0.04
    mode      = 'test'  # set train or valid
    n_num     = 10

    # io varaiables
    #root_path = '../training_data/n{}/run{}/'.format(nparticle, n_num)
    root_path = '../test_data/n{}/run{}/'.format(nparticle, n_num)
    write_filename = root_path + 'nparticle{}_new_nsim_rho0.1_T{}_pos_{}_sampled.pt'.format(nparticle, temp, mode)

    filename = sorted(glob.glob(root_path + '*' +'id' +'*.pt'), key=keyFunc)
    qp_list_app = qp_list_combine(filename)
    torch.save(qp_list_app, write_filename)
    print('combined files', qp_list_app.shape)
