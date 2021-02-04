import torch
from MD_paramaters import MD_parameters
from phase_space.phase_space import phase_space

class data_io:

    def hamiltonian_dataset(self, filename, ratio : float):

        q_list, p_list = torch.load(filename)

        # shuffle
        g = torch.Generator()
        g.manual_seed(MD_parameters.seed)

        idx = torch.randperm(q_list.shape[0], generator=g)

        q_list_shuffle_ = q_list[idx]
        p_list_shuffle_ = p_list[idx]

        q_list_shuffle = q_list_shuffle_[:MD_parameters.nsamples]
        p_list_shuffle = p_list_shuffle_[:MD_parameters.nsamples]

        try:
            assert q_list_shuffle.shape == p_list_shuffle.shape
        except:
             raise Exception('does not have shape method or shape differs')

        # print('after shuffle')
        # print(q_list_shuffle, p_list_shuffle)

        init_pos = torch.unsqueeze(q_list_shuffle, dim=1) #  nsamples  X 1 X nparticle X  DIM
        init_vel = torch.unsqueeze(p_list_shuffle, dim=1)  #  nsamples  X 1 X nparticle X  DIM
        init = torch.cat((init_pos, init_vel), dim=1)  # nsamples X 2 X nparticle X  DIM

        train_data = init[:int(ratio*init.shape[0])]
        valid_data = init[int(ratio*init.shape[0]):]

        return train_data, valid_data

    def phase_space2label(self, qp_list, linear_integrator, noML_hamiltonian):

        nsamples, qnp, nparticles, DIM = qp_list.shape

        q_list = qp_list[:,0]
        p_list = qp_list[:,1]
        # print('phase_space2label input',q_list.shape, p_list.shape)
        # print('nsamples',nsamples)

        _phase_space = phase_space()
        _phase_space.set_q(q_list)
        _phase_space.set_p(p_list)

        # print('===== state at short time step 0.01 =====')
        nsamples_cur = nsamples # train or valid
        tau_cur = MD_parameters.tau_short
        MD_iterations = int(MD_parameters.tau_long / tau_cur)

        print('prepare labels nsamples_cur, tau_cur, MD_iterations')
        print(nsamples_cur, tau_cur, MD_iterations)
        label = linear_integrator.integrate( noML_hamiltonian, _phase_space, MD_iterations, nsamples_cur, tau_cur)

        return label