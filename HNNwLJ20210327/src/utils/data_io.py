import torch
import pickle
import gzip
import os
from MC_parameters import MC_parameters
from MD_parameters import MD_parameters

class data_io:

    def __init__(self,root_dir_name):
        ''' set up the root directory for all data '''
        self.root_dir = root_dir_name

    def read_init_qp(self, mode):
        ''' given mode ( train or valid or test), read the files at different temp
            and then combine them

            returns
            torch.tensor of qp_list

            use in MD_sample.py and dataset.py at HNN folder
        '''

        file_format =  self.root_dir + 'nparticle' + str(MC_parameters.nparticle) + '_new_nsim' + '_rho{}_T{}_pos_' + str(mode) + '_sampled.pt'

        q_list = None
        p_list = None

        for i, temp in enumerate(MD_parameters.temp_list):

            print('temp', temp)
            q_curr, p_curr = torch.load(file_format.format(MC_parameters.rho, temp))

            if i == 0:
                q_list = q_curr
                p_list = p_curr
            else:
                q_list = torch.cat((q_list, q_curr))
                p_list = torch.cat((p_list, p_curr))

            assert q_list.shape == p_list.shape

        return (q_list, p_list)

    def write_init_qp(self, qp_list, filename):
        ''' write or append to filename for qp_list

        Parameters
        ----------
        filename : string
        qp_list : torch.tensor
                  tensor of (q,p) states
                  shape is [(q,p), (new_mcs x mc step), nparticle, DIM]

        use in MC_sample.py
        '''

        file_format = self.root_dir + filename
        torch.save(qp_list, file_format)

    # didn't write qp end pts for training....  code for label data is in dataset.py at HNN folder

    # def read_endpts_qp(self, filename):
    #     ''' given a filename, read the qp end pts for training
    #         returns torch tensor pair of qp_list '''
    #
    #     with gzip.open( filename, 'rb') as handle: # overwrites any existing file
    #         qp_list = pickle.load(handle)
    #
    #         return qp_list
    #
    # def write_endpts_qp(self, filename, qp_list_pair):
    #     ''' write or append to filename for qp_list
    #         mode = 'w' or 'a' '''
    #
    #     with gzip.open( filename, 'wb') as handle: # overwrites any existing file
    #         pickle.dump(qp_list_pair, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #         handle.close()

    def read_trajectory_qp(self,tmp_filename, no_file):
        ''' given a temporary filename, read the qp paired pts trajectory for testing or gold standard

        returns
        torch tensor of qp_list

        use in MD_sample.py
        '''

        with gzip.open( tmp_filename + '_{}.pt'.format(no_file) , 'rb') as handle: # overwrites any existing file
            qp_list = pickle.load(handle)

            return qp_list

    def write_trajectory_qp(self, tmp_filename, no_file, qp_trajectory):
        ''' write or append to temporary filename for qp_trajectory

        Parameters
        ----------
        tmp_filename : string
        no_file : int
                i th saved file
        qp_trajectory : torch.tensor
                  tensor of (q,p) states
                  shape is [iteration_batch, (q, p), nsamples, nparticle, DIM]

        returns
        torch tensor
        qp paired pts trajectory

        use in MD_sample.py
        '''

        with gzip.open( tmp_filename + '_{}.pt'.format(no_file), 'wb') as handle: # overwrites any existing file
            pickle.dump(qp_trajectory, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()


    def read_crash_qp(self, filename):
        ''' given a filename, read the qp pts for retraining

            returns
            torch tensor pair of qp_list '''

        with gzip.open( filename , 'rb') as handle: # overwrites any existing file
            qp_list = pickle.load(handle)

            return qp_list

    def write_crash_qp(self, filename, crash_qp):
        ''' write or append to filename  for retraining '''

        with gzip.open( filename, 'wb') as handle: # overwrites any existing file
            pickle.dump(crash_qp, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()