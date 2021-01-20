import matplotlib.pyplot as plt
import torch
import shutil
import time

class MD_learner:

    def __init__(self,linear_integrator, noML_hamiltonian, pair_wise_HNN, **state):

        self.linear_integrator = linear_integrator
        self.noML_hamiltonian =noML_hamiltonian
        self.pair_wise_HNN = pair_wise_HNN
        self._state = state

        self._MLP = state['MLP'].to(state['_device'])
        self._opt = state['opt']
        self._loss = state['loss']

        self._current_epoch = 1
        # initialize best model
        self._best_validation_loss = float('inf')


    def phase_space2label(self, MD_integrator, noML_hamiltonian):
        label = MD_integrator.integrate(noML_hamiltonian)
        return label

    # phase_space consist of minibatch data
    def trainer(self, filename):

        # print('===== load initial data =====')
        q_list, p_list = self._state['phase_space'].read(filename, nsamples= self._state['nsamples_label'])

        self._state['phase_space'].set_q(q_list)
        self._state['phase_space'].set_p(p_list)

        # print('===== state at short time step 0.01 =====')
        self._state['nsamples_cur'] = self._state['nsamples_label']
        self._state['tau_cur'] = self._state['tau_short']
        self._state['MD_iterations'] = int(self._state['tau_long']/self._state['tau_cur'])

        q_list_label, p_list_label = self.phase_space2label(self.linear_integrator(**self._state), self.noML_hamiltonian)

        # to prepare data at large time step, need to change tau and iterations
        # tau = large time step 0.1 and 1 step
        # print('===== state at large time step 0.1 =====')

        self._state['nsamples_cur'] = self._state['nsamples_ML']
        self._state['tau_cur'] = self._state['tau_long']  # tau = 0.1
        self._state['MD_iterations'] = int(self._state['tau_long']/self._state['tau_cur'])

        pairwise_hnn = self.pair_wise_HNN(self.noML_hamiltonian, self._MLP, **self._state)
        pairwise_hnn.train()

        n_batches = q_list.shape[0]
        text = ''

        for e in range(self._state['nepochs']):

            # shuffle based on epoch
            g = torch.Generator()
            # g.manual_seed(e)

            # all same shuffle idx each epoch
            idx = torch.randperm(q_list.shape[0], generator=g)
            # print('each epoch idx', idx)
            q_list_shuffle, p_list_shuffle = q_list[idx], p_list[idx]

            train_loss = 0
            start = time.time()

            for i in range(n_batches):

                q_list_batch, p_list_batch = q_list_shuffle[i], p_list_shuffle[i]
                q_list_batch = torch.unsqueeze(q_list_batch, dim=0).to(self._state['_device'])
                p_list_batch = torch.unsqueeze(p_list_batch, dim=0).to(self._state['_device'])

                q_label_batch, p_label_batch = q_list_label[idx[i]], p_list_label[idx[i]]
                q_label_batch = torch.unsqueeze(q_label_batch, dim=0).to(self._state['_device'])
                p_label_batch = torch.unsqueeze(p_label_batch, dim=0).to(self._state['_device'])

                label = (q_label_batch, p_label_batch)

                self._state['phase_space'].set_q(q_list_batch)
                self._state['phase_space'].set_p(p_list_batch)


                # print('======= train combination of MD and ML =======')
                q_pred, p_pred = self.linear_integrator(**self._state).integrate(pairwise_hnn)
                q_pred = q_pred.to(self._state['_device']); p_pred = p_pred.to(self._state['_device'])

                prediction = (q_pred, p_pred)

                loss = self._loss(prediction, label)

                self._opt.zero_grad()  # defore the backward pass, use the optimizer object to zero all of the gradients for the variables
                loss.backward()  # backward pass : compute gradient of the loss wrt model parameters

                self._opt.step()

                train_loss += loss.item()  # get the scalar output
                # print('loss each batch',loss.item())

            train_loss_avg = train_loss / n_batches
            end = time.time()
            print('{} epoch:'.format(e),train_loss_avg, ' time:', end-start)

            self.record_best(train_loss_avg, 'nsamples{}_tau{}_lr{}_h{}_checkpoint.pth'.format(self._state['nsamples_label'], self._state['tau_cur'],
                                                     self._opt.param_groups[0]['lr'], self._state['n_hidden']))

            text = text + str(e) + ' ' + str(train_loss_avg)  + '\n'
            with open('nsamples{}_tau{}_loss.txt'.format(self._state['nsamples_label'],self._state['tau_cur']), 'w') as fp:
                fp.write(text)
            fp.close()

            self._current_epoch += 1


    def record_best(self, validation_loss, filename):

        is_best = validation_loss < self._best_validation_loss
        self._best_validation_loss = min(validation_loss, self._best_validation_loss)

        torch.save(({
                'epoch': self._current_epoch,
                'model_state_dict' : self._MLP.state_dict(),
                'best_validation_loss' : self._best_validation_loss,
                'optimizer': self._opt.state_dict()
                }, is_best), filename)

        if is_best:
            shutil.copyfile(filename, 'nsamples{}_tau{}_lr{}_h{}_checkpoint_best.pth'.format(self._state['nsamples_label'], self._state['tau_cur'],
                                                     self._opt.param_groups[0]['lr'], self._state['n_hidden']))


    def pred_qnp(self, filename):

        # load the model checkpoint
        checkpoint = torch.load('nsamples{}_tau{}_lr{}_h{}_checkpoint.pth'.format(
            self._state['nsamples_label'], self._state['tau_long'], self._opt.param_groups[0]['lr'], self._state['n_hidden']))[0]

        # load model weights state_dict
        self._MLP.load_state_dict(checkpoint['model_state_dict'])
        print('Previously trained model weights state_dict loaded...')
        self._opt.load_state_dict(checkpoint['optimizer'])
        print('Previously trained optimizer state_dict loaded...')

        # initial data
        q_list, p_list = self._state['phase_space'].read(filename, nsamples=self._state['nsamples_label'])
        # print(q_list, p_list)

        self._state['phase_space'].set_q(torch.unsqueeze(q_list[0], dim=0).to(self._state['_device']))
        self._state['phase_space'].set_p(torch.unsqueeze(p_list[0], dim=0).to(self._state['_device']))

        self._state['nsamples_cur'] = self._state['nsamples_ML']
        self._state['tau_cur'] = self._state['tau_long']  # tau = 0.1
        self._state['MD_iterations'] = int(self._state['tau_long']/self._state['tau_cur'])

        pairwise_hnn = self.pair_wise_HNN(self.noML_hamiltonian, self._MLP, **self._state)
        pairwise_hnn.eval()

        q_pred, p_pred = self.linear_integrator(**self._state).integrate(pairwise_hnn)

        self._state['tau_cur'] = self._state['tau_short']

        self._state['phase_space'].set_q(torch.unsqueeze(q_list[0], dim=0))
        self._state['phase_space'].set_p(torch.unsqueeze(p_list[0], dim=0))

        q_truth, p_truth = self.phase_space2label(self.linear_integrator(**self._state), self.noML_hamiltonian)

        print('predict',q_pred, p_pred)
        print('truth', q_truth, p_truth)

        return q_pred, p_pred



    def step(self,phase_space,pb,tau):
        pairwise_hnn.eval()
        q_list_predict, p_list_predict = self.linear_integrator.integrate(**self._state)
        return q_list_predict,p_list_predict
