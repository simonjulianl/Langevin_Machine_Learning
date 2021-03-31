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
    ''' change parameters and root_path( for traininig data or test data ) you need to run this code '''

    # paramters
    nparticle = 2
    mode      = 'train'  # set train or valid or test
    n_num     = 11

    temp      = 0.04     # use only test data

    # io varaiables for training data
    root_path = '../training_data/n{}/run{}/'.format(nparticle, n_num)
    init_filename = root_path + 'nparticle{}_new_nsim_rho0.1_allT_{}_sampled.pt'.format(nparticle, mode)
    write_filename = root_path + '../nparticle{}_rho0.1_allT_{}_sampled.pt'.format(nparticle, mode)

    # # io varaiables for test data
    # root_path = '../test_data/n{}/run{}/'.format(nparticle, n_num)
    # init_filename = root_path + 'nparticle{}_new_nsim_rho0.1_T{}_pos_test_sampled.pt'.format(nparticle, temp)
    # write_filename = root_path + 'nparticle{}_rho0.1_T{}_test_sampled.pt'.format(nparticle, temp)

    # returns the list of files with their full path
    filename = sorted(glob.glob(root_path + '*' +'_id' +'*.pt'), key=keyFunc)
    qp_list_app = qp_list_combine(filename)

    # load init q, p
    qp_init = torch.load(init_filename)
    qp_init = torch.unsqueeze(qp_init, dim = 0)

    # concatenate qp_init and append strike qp state
    qp_list = torch.cat((qp_init, qp_list_app))

    torch.save(qp_list, write_filename)
    print('combined files', qp_list.shape)
