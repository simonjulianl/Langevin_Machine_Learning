import matplotlib.pyplot as plt
import torch

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
        q_list, p_list = state['phase_space'].read(filename, nsamples=state['nsamples_label'])
        # print(q_list, p_list)

        state['phase_space'].set_q(q_list)
        state['phase_space'].set_p(p_list)

        # print('===== state at short time step 0.01 =====')
        state['nsamples_cur'] = state['nsamples_label']
        state['tau_cur'] = state['tau_short']
        state['MD_iterations'] = int(state['tau_long']/state['tau_cur'])

        q_list_label, p_list_label = self.phase_space2label(self.linear_integrator(**state), self.noML_hamiltonian)

        # to prepare data at large time step, need to change tau and iterations
        # tau = large time step 0.1 and 1 step
        # print('===== state at large time step 0.1 =====')

        state['nsamples_cur'] = state['nsamples_ML']
        state['tau_cur'] = state['tau_long']  # tau = 0.1
        state['MD_iterations'] = int(state['tau_long']/state['tau_cur'])
        state['MLP'] = state['MLP'].to(state['_device'])

        pairwise_hnn = self.pair_wise_HNN(self.noML_hamiltonian, state['MLP'], **state)
        pairwise_hnn.train()
        opt = state['opt']

        loss_ = []
        n_batches = q_list.shape[0]

        for e in range(state['nepochs']):

            # shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(e)

            idx = torch.randperm(q_list.shape[0], generator=g)
            q_list_shuffle, p_list_shuffle = q_list[idx], p_list[idx]

            train_loss = 0

            for i in range(n_batches):

                print('batch',i)
                q_list_batch, p_list_batch = q_list_shuffle[idx[i]], p_list_shuffle[idx[i]]
                q_list_batch = torch.unsqueeze(q_list_batch, dim=0).to(state['_device']).requires_grad_(True)
                p_list_batch = torch.unsqueeze(p_list_batch, dim=0).to(state['_device']).requires_grad_(True)

                state['phase_space'].set_q(q_list_batch)
                state['phase_space'].set_p(p_list_batch)

                # print('======= train combination of MD and ML =======')
                prediction = self.linear_integrator(**state).integrate(pairwise_hnn)

                q_label_batch, p_label_batch = q_list_label[idx[i]], p_list_label[idx[i]]
                q_label_batch = torch.unsqueeze(q_label_batch, dim=0).to(state['_device'])
                p_label_batch = torch.unsqueeze(p_label_batch, dim=0).to(state['_device'])

                label = (q_label_batch, p_label_batch)

                loss = state['loss'](prediction, label)

                opt.zero_grad()  # defore the backward pass, use the optimizer object to zero all of the gradients for the variables
                loss.backward()  # backward pass : compute gradient of the loss wrt model parameters
                train_loss += loss.item()  # get the scalar output
                opt.step()

            train_loss_avg = train_loss / n_batches
            print(train_loss_avg)
            loss_.append(train_loss_avg)

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