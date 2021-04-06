import torch
import pickle
import gzip
import os

class data_io:

    def read_init_qp(self, filename):
        '''
            returns
            torch.tensor of qp_list

            shape of q_list is [nsamples,nparticles,DIM]
            shape of p_list is [nsamples,nparticles,DIM]
        '''
        qp_list = torch.load(filename)

        return qp_list


    def write_init_qp(self, filename, qp_list):
        ''' write or append to filename for qp_list

        Parameters
        ----------
        filename : string
        qp_list : torch.tensor q_list shape is [nsamples, nparticle, DIM]
                  tensor of (q,p) states
                  shape is [(q,p), (new_mcs x mc step), nparticle, DIM]

        use in MC_sampler.py
        '''
        torch.save(qp_list, filename)


    def read_trajectory_qp(self, filename):
        ''' given a temporary filename, read the qp paired pts trajectory for testing or gold standard

        returns
        torch tensor of qp_list

        use in MD_sampler.py
        '''

        qp_list = torch.load(filename)

        return qp_list

    def write_trajectory_qp(self, filename, qp_trajectory):
        ''' write or append to temporary filename for qp_trajectory

        Parameters
        ----------
        filename : string
        qp_trajectory : torch.tensor
                  tensor of (q,p) states
                  shape is [trajectory length, (q, p), nsamples, nparticle, DIM]

        returns
        torch tensor
        qp paired pts trajectory

        use in MD_sampler.py
        '''

        torch.save(qp_trajectory, filename)