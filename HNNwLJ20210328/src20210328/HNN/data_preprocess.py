import torch
from utils.data_io              import data_io
from parameters.MD_parameters   import MD_parameters

class data_preprocess:

    ''' data_preprocess class to help preprocess dataset
        - combine q p list that is different condition ( e.g temp)
        - label data '''

    _obj_count = 0

    def __init__(self, linear_integrator, root_dir_name):

        data_preprocess._obj_count += 1
        assert (data_preprocess._obj_count == 1),type(self).__name__ + " has more than one object"

        self.data_io_obj = data_io(root_dir_name)
        self.linear_integrator_obj = linear_integrator

    def qp_list_combine(self, init_filename, mode):
        ''' function to combine qp list at different temp

        qp_list : torch.tensor
                shape is [ (q,p), nsamples, nparticle, DIM ]

        mode : str

        return : list
                add qp_list to the list
        '''
        qp_list_app = []
        for i, temp in enumerate(MD_parameters.temp_list):

            # qp_list shape is [(q,p), nsamples, nparticle, DIM]
            qp_list = self.data_io_obj.read_init_qp( init_filename + 'T{}_pos_'.format(temp) + str(mode) + '_sampled.pt')
            qp_list_app.append(qp_list)

        qp_list_app = torch.cat(qp_list_app, dim=1) # concat along nsampless

        return qp_list_app

    def phase_space2label(self, qp_list, noMLhamiltonian, phase_space, tau_short, niter_tau_short, append_strike):

        phase_space.set_q(qp_list[0])
        phase_space.set_p(qp_list[1])

        qp_list_label = self.linear_integrator_obj.nsteps( noMLhamiltonian, phase_space, tau_short, niter_tau_short, append_strike)
        qp_list_label = torch.stack(qp_list_label)

        return qp_list_label