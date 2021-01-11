
class MD_learner:

    def __init__(self,linear_integrator, NoML_hamiltonian, pair_wise_HNN):

        self.linear_integrator = linear_integrator
        self.NoML_hamiltonian =NoML_hamiltonian
        self.pair_wise_HNN = pair_wise_HNN

    def phase_space2label(self, MD_integrator, noML_hamiltonian):
        label = MD_integrator.integrate(noML_hamiltonian)
        return label

    # phase_space consist of minibatch data
    # pb is boundary condition
    def train(self, filename, **state):

        # load initial data
        q_list, p_list = state['phase_space'].read(filename, nsamples=state['N'])

        state['phase_space'].set_q(q_list)
        state['phase_space'].set_p(p_list)

        label = self.phase_space2label(self.linear_integrator(**state), self.NoML_hamiltonian)

        self.pair_wise_HNN.train()
        opt = state['opt']

        state['tau'] = state['tau'] * state['iterations']  # tau = 0.1
        state['iterations'] = int(state['tau'] * state['iterations'])  # 1 step

        for e in range(state['nepochs']):

            state['phase_space'].set_q(q_list)
            state['phase_space'].set_p(p_list)

            prediction = self.linear_integrator(**state).integrate(self.pair_wise_HNN)


            loss = state['loss'](prediction, label)

            opt.zero_grad()  # defore the backward pass, use the optimizer object to zero all of the gradients for the variables
            loss.backward()  # backward pass : compute gradient of the loss wrt model parameters
            train_loss = loss.item()  # get the scalar output
            opt.step()

            print('epoch loss ', e, train_loss)

