import torch

class data_io:

    # standardize shape of qp_list is [nsamples, (q, p), trajectory length, nparticle, DIM]
    # standardize shape of q_list  is [nsamples,         trajectory length, nparticle, DIM]
    # ================================================
    @staticmethod
    def read_trajectory_qp(filename):
        ''' given a filename, read the qp paired pts trajectory

        returns
        load the dictionary and then return all
        shape of qp_list is [nsamples, (q, p), trajectory length, nparticle, DIM]
        '''

        data = torch.load(filename)

        qp_list = data['qp_trajectory']
        tau_long  = data['tau_long']
        tau_short = data['tau_short']

        return qp_list, tau_long, tau_short

    # ================================================
    @staticmethod
    def write_trajectory_qp(filename, qp_trajectory, tau_short = -1, tau_long = -1):
        ''' write temporary filename for qp_trajectory

        Parameters
        ----------
        filename : string
        qp_trajectory : torch.tensor
                  tensor of (q,p) states
                  shape is [nsamples, (q, p), trajectory length, nparticle, DIM]
        tau_short tau_long : float or int
                  default is negative values for MC output
                  positive values for MD output

        returns
        save multiple components, organize them in a dictionary
        '''

        data = { 'qp_trajectory':qp_trajectory, 'tau_short':tau_short, 'tau_long': tau_long }

        torch.save(data, filename)
