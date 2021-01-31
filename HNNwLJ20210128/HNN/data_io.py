import torch

class data_io:

    def __init__(self, filename):

        self._filename =filename

    def hamiltonian_dataset(self, ratio : float):

        phase_space = self._state['phase_space'].read(self._filename, nsamples=self._state['nsamples_label'])
        q_list, p_list = phase_space[0][:self._state['nsamples_label']], phase_space[1][:self._state['nsamples_label']]
        # print('nsamples')
        # print(q_list,p_list)

        assert q_list.shape == p_list.shape  # shape of p and q must be the same

        # shuffle
        g = torch.Generator()
        # g.manual_seed(seed)

        idx = torch.randperm(q_list.shape[0], generator=g)
        # print(idx)

        q_list_shuffle = q_list[idx]
        p_list_shuffle = p_list[idx]

        # print(q_list_shuffle,p_list_shuffle)

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

        self._state['phase_space'].set_q(q_list)
        self._state['phase_space'].set_p(p_list)

        # print('===== state at short time step 0.01 =====')
        self._state['nsamples_cur'] = nsamples # train or valid
        self._state['tau_cur'] = self._state['tau_short']
        self._state['MD_iterations'] = int(self._state['tau_long']/self._state['tau_cur'])

        label = linear_integrator(**self._state).integrate(noML_hamiltonian)

        return label