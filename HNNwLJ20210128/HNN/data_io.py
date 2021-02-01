import torch
from HNNwLJ20210128.parameters.MD_paramaters import MD_parameters

class data_io:

    def __init__(self, phase_space, filename):

        self._phase_space = phase_space
        self._filename = filename

    def hamiltonian_dataset(self, ratio : float):

        _phase_space = self._phase_space.read(self._filename, nsamples = MD_parameters.select_nsamples)
        q_list, p_list = _phase_space[0][:MD_parameters.select_nsamples], _phase_space[1][:MD_parameters.select_nsamples]

        # print('before shuffle')
        # print(q_list.shape, p_list.shape)

        assert q_list.shape == p_list.shape  # shape of p and q must be the same

        # shuffle
        g = torch.Generator()
        g.manual_seed(MD_parameters.seed)

        idx = torch.randperm(q_list.shape[0], generator=g)
        # print('idx', idx)

        q_list_shuffle = q_list[idx]
        p_list_shuffle = p_list[idx]

        # print('after shuffle')
        # print(q_list_shuffle, p_list_shuffle)

        init_pos = torch.unsqueeze(q_list_shuffle, dim=1) #  nsamples  X 1 X nparticle X  DIM
        init_vel = torch.unsqueeze(p_list_shuffle, dim=1)  #  nsamples  X 1 X nparticle X  DIM
        init = torch.cat((init_pos, init_vel), dim=1)  # nsamples X 2 X nparticle X  DIM

        train_data = init[:int(ratio*init.shape[0])]
        valid_data = init[int(ratio*init.shape[0]):]

        return train_data, valid_data

    def phase_space2label(self, qnp_list, linear_integrator, noML_hamiltonian):

        nsamples, qnp, nparticles, DIM = qnp_list.shape

        q_list = qnp_list[:,0]
        p_list = qnp_list[:,1]
        # print('phase_space2label input',q_list,p_list)
        # print('nsamples',nsamples)

        self._phase_space.set_q(q_list)
        self._phase_space.set_p(p_list)

        # print('===== state at short time step 0.01 =====')
        nsamples_cur = nsamples # train or valid
        tau_cur = MD_parameters.tau_short
        MD_iterations = int(MD_parameters.tau_long / tau_cur)

        print('prepare labels nsamples_cur, tau_cur, MD_iterations')
        print(nsamples_cur, tau_cur, MD_iterations)
        label = linear_integrator.integrate( noML_hamiltonian, self._phase_space, MD_iterations, nsamples_cur, tau_cur)

        return label