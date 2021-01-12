import matplotlib.pyplot as plt

class MD_learner:

    def __init__(self,linear_integrator, noML_hamiltonian, pair_wise_HNN):

        self.linear_integrator = linear_integrator
        self.noML_hamiltonian =noML_hamiltonian
        self.pair_wise_HNN = pair_wise_HNN

    def phase_space2label(self, MD_integrator, noML_hamiltonian):
        label = MD_integrator.integrate(noML_hamiltonian)
        return label

    # phase_space consist of minibatch data
    def trainer(self, filename, **state):

        # print('===== load initial data =====')
        q_list, p_list = state['phase_space'].read(filename, nsamples=state['nsamples'])

        state['phase_space'].set_q(q_list)
        state['phase_space'].set_p(p_list)

        # print('===== state at short time step 0.01 =====')
        state['tau_cur'] = state['tau_short']
        state['MD_iterations'] = int(state['tau_long']/state['tau_cur'])
        label = self.phase_space2label(self.linear_integrator(**state), self.noML_hamiltonian)

        # to prepare data at large time step, need to change tau and iterations
        # tau = large time step 0.1 and 1 step
        state['tau_cur'] = state['tau_long']  # tau = 0.1
        state['MD_iterations'] = int(state['tau_long']/state['tau_cur'])

        pairwise_hnn = self.pair_wise_HNN(self.noML_hamiltonian, state['MLP'], **state)
        pairwise_hnn.train()
        opt = state['opt']

        loss_ = []

        for e in range(state['nepochs']):

            state['phase_space'].set_q(q_list)
            state['phase_space'].set_p(p_list)

            # print('======= train combination of MD and ML =======')
            prediction = self.linear_integrator(**state).integrate(pairwise_hnn)

            loss = state['loss'](prediction, label)

            opt.zero_grad()  # defore the backward pass, use the optimizer object to zero all of the gradients for the variables
            loss.backward()  # backward pass : compute gradient of the loss wrt model parameters
            train_loss = loss.item()  # get the scalar output
            opt.step()

            #print('epoch loss ', e, train_loss)
            loss_.append(train_loss)

        plt.xlabel('epoch', fontsize=20)
        plt.ylabel('loss', fontsize=20)
        plt.plot(loss_, linewidth=2)
        plt.grid()
        # plt.show()


        # # do one step velocity verlet without ML
        # print('do one step velocity verlet without ML')
        # print(state)
        #
        # state['phase_space'].set_q(q_list)
        # state['phase_space'].set_p(p_list)
        #
        # prediction_noML = label
        #
        # print('prediction with   ML', prediction)
        # print('prediction with noML', prediction_noML)
        #
        # q_pred, p_pred = prediction
        # q_label, p_label = prediction_noML
        #
        # now_loss = (q_pred - q_label) ** 2 + (p_pred - p_label) ** 2
        # now_loss = (now_loss).sum()
        # train_loss = state['loss'](prediction, label)
        # print('previous loss', train_loss.item())  # label at short time step 0.01
        # print('now      loss', now_loss.item())  # label at large time step 0.1


    def step(self,phase_space,pb,tau):
        pairwise_hnn.eval()
        q_list_predict, p_list_predict = self.linear_integrator.integrate(**state)
        return q_list_predict,p_list_predict