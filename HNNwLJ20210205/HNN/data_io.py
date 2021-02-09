import torch
from MD_paramaters import MD_parameters

class data_io:

    _obj_count = 0

    def __init__(self):

        super().__init__()

        data_io._obj_count += 1
        assert(data_io._obj_count == 1), type(self).__name__ + ' has more than one object'

    def hamiltonian_dataset(self, filename):

        q_list, p_list = torch.load(filename)

        # shuffle
        g = torch.Generator()
        g.manual_seed(MD_parameters.seed)

        idx = torch.randperm(q_list.shape[0], generator=g)

        q_list_shuffle = q_list[idx]
        p_list_shuffle = p_list[idx]

        try:
            assert q_list_shuffle.shape == p_list_shuffle.shape
        except:
             raise Exception('does not have shape method or shape differs')

        # print('after shuffle')
        # print(q_list_shuffle, p_list_shuffle)

        init_pos = torch.unsqueeze(q_list_shuffle, dim=1) #  nsamples  X 1 X nparticle X  DIM
        init_vel = torch.unsqueeze(p_list_shuffle, dim=1)  #  nsamples  X 1 X nparticle X  DIM
        init = torch.cat((init_pos, init_vel), dim=1)  # nsamples X 2 X nparticle X  DIM

        return init

    def phase_space2label(self, qp_list, linear_integrator, phase_space, noML_hamiltonian):

        nsamples, qnp, nparticles, DIM = qp_list.shape

        q_list = qp_list[:,0]
        p_list = qp_list[:,1]
        # print('phase_space2label input',q_list.shape, p_list.shape)
        # print('nsamples',nsamples)

        phase_space.set_q(q_list)
        phase_space.set_p(p_list)

        # print('===== state at short time step 0.01 =====')
        nsamples_cur = nsamples # train or valid
        tau_cur = MD_parameters.tau_short
        MD_iterations = int(MD_parameters.tau_long / tau_cur)

        print('prepare labels nsamples_cur, tau_cur, MD_iterations')
        print(nsamples_cur, tau_cur, MD_iterations)
        label = linear_integrator.step( noML_hamiltonian, phase_space, MD_iterations, nsamples_cur, tau_cur)

        return label