import torch
from utils.data_io import data_io

class data_handler:

    ''' data_handler class to help load and shuffle data for ML '''

    _obj_count = 0

    def __init__(self):

        data_handler._obj_count += 1
        assert(data_handler._obj_count == 1), type(self).__name__ + ' has more than one object'

        self.qp_list_input = None
        self.qp_list_label = None
        self.tau_long      = None
        self.tau_short     = None

    def load(self, load_filename):
        ''' function to load file ( train or valid or test)

        parameter
        qp_list : torch.tensor
                shape is [niter, 2, nsamples, nparticle, DIM], here 2 is (q, p)
                niter : initial and append strike iter

        return : torch.tensor
                each shape is [2, nsamples, nparticle, DIM]
        '''
        qp_list, tau_long, tau_short = data_io.read_trajectory_qp(load_filename)

        self.qp_list_input = qp_list[0, :]  # initial
        self.qp_list_label = qp_list[1, :]  # append strike
        self.tau_long      = tau_long
        self.tau_short     = tau_short


    def _shuffle(self, qp_list_input, qp_list_label):
        ''' function to shuffle data ( use train or valid,  not use test)

        parameter
        qp_list_input : torch.tensor
                shape is [2, nsamples, nparticle, DIM], here 2 is (q, p)

        return : torch.tensor
                each shape is [2, nsamples, nparticle, DIM]
        '''

        idx = torch.randperm(qp_list_input.shape[1])  # nsamples

        qp_list_input_shuffle = qp_list_input[:,idx]
        qp_list_label_shuffle = qp_list_label[:,idx]

        try:
            assert qp_list_input_shuffle.shape == qp_list_label_shuffle.shape
        except:
             raise Exception('does not have shape method or shape differs')

        return qp_list_input_shuffle, qp_list_label_shuffle

