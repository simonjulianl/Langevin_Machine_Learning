import torch

class data_io:

    @staticmethod
    def read_init_qp(filename):
        '''
            returns
            torch.tensor of qp_list

            shape of q_list is [nsamples,nparticles,DIM]
            shape of p_list is [nsamples,nparticles,DIM]
        '''

        data = torch.load(filename)

        qp_list = data['qp_trajectory']
        tau_long  = data['tau_long']
        tau_short = data['tau_short']

        return qp_list, tau_long, tau_short


    @staticmethod
    def write_init_qp(filename, qp_list):
        ''' write or append to filename for qp_list

        Parameters
        ----------
        filename : string
        qp_list : torch.tensor q_list shape is [nsamples, nparticle, DIM]
                  tensor of (q,p) states
                  shape is [(q,p), (new_mcs x mc step), nparticle, DIM]

        returns
        save multiple components, organize them in a dictionary

        use in MC_sampler.py
        '''

        data = {'qp_trajectory': qp_list, 'tau_short': -1.0, 'tau_long': -1.0}

        torch.save(data, filename)


    @staticmethod
    def read_trajectory_qp(filename):
        ''' given a temporary filename, read the qp paired pts trajectory for testing or gold standard

        returns
        load the dictionary and then return all

        use in MD_sampler.py
        '''

        data = torch.load(filename)

        qp_list = data['qp_trajectory']
        tau_long  = data['tau_long']
        tau_short = data['tau_short']

        return qp_list, tau_long, tau_short

    @staticmethod
    def write_trajectory_qp(filename, qp_trajectory, tau_short, tau_long):
        ''' write or append to temporary filename for qp_trajectory

        Parameters
        ----------
        filename : string
        qp_trajectory : torch.tensor
                  tensor of (q,p) states
                  shape is [trajectory length, (q, p), nsamples, nparticle, DIM]

        returns
        save multiple components, organize them in a dictionary

        use in MD_sampler.py
        '''

        data = { 'qp_trajectory':qp_trajectory, 'tau_short':tau_short, 'tau_long': tau_long }

        torch.save(data, filename)
