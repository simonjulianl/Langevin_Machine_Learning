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
            shape is [ 1, (q,p), nsamples, nparticle, DIM ] combined along different temp

    return : list
            add qp_list to the list
    '''

    qp_list_app = []

    for i in range(len(filename)):

        print(filename[i])
        data = torch.load(filename[i])
        qp_list = data['qp_trajectory']

        qp_list_app.append(qp_list)

    #shape is [nsamples, trajectory length, (q, p), nparticle, DIM]
    qp_list_app = torch.cat(qp_list_app, dim=0) # along nsamples

    return qp_list_app

if __name__ == '__main__':

    ''' combine initial q, p at different temperature for training  '''
    # paramters
    nparticle = 2
    mode      = 'train'  # set train or valid

    # io varaiables
    root_path = '../init_config/n2/'
    filename = sorted(glob.glob(root_path + 'n{}_nsim_rho0.1T'.format(nparticle) + '*' + '_{}_sampled.pt'.format(mode)), key=keyFunc)
    write_filename = root_path + 'n{}_nsim_rho0.1allT_{}_sampled.pt'.format(nparticle, mode)
   
    qp_list_app = qp_list_combine(filename)

    data = {'qp_trajectory': qp_list_app,  'tau_short': -1.0, 'tau_long': -1.0}

    torch.save(data, write_filename)
