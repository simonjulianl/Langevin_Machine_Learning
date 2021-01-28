from .dataset_split import dataset_split
import torch
import shutil
import time
import os

class MD_learner:

    def __init__(self,linear_integrator, noML_hamiltonian, pair_wise_HNN, filename, **state):

        self.linear_integrator = linear_integrator
        self.noML_hamiltonian =noML_hamiltonian
        self.pair_wise_HNN = pair_wise_HNN

        self._state = state
        self._train_data, self._valid_data, self._train_label, self._valid_label = self.load_data(filename)
        self._filename = filename

        self._MLP = state['MLP'].to(state['_device'])
        self._opt = state['opt']
        self._loss = state['loss']

        self._current_epoch = 1
        # initialize best models
        self._best_validation_loss = float('inf')

    def load_data(self, filename):

        dataset_obj = dataset_split(filename, **self._state)

        # nsamples X qnp X nparticle X  DIM
        train_data, valid_data = dataset_obj.hamiltonian_dataset(ratio=0.7)

        # qnp x iterations x nsamples x  nparticle x DIM
        train_label = dataset_obj.phase_space2label(train_data, self.linear_integrator, self.noML_hamiltonian)
        valid_label = dataset_obj.phase_space2label(valid_data, self.linear_integrator, self.noML_hamiltonian)

        return train_data, valid_data, train_label, valid_label

    # phase_space consist of minibatch data
    def train_valid_epoch(self):

        # print('===== load initial train data =====')
        q_train = self._train_data[:,0]; p_train = self._train_data[:,1]

        # print('===== label train data =====')
        q_train_label = self._train_label[0]; p_train_label = self._train_label[1]
        q_train_label = q_train_label[-1]; p_train_label = p_train_label[-1] # only take the last from the list

        assert q_train.shape == q_train_label.shape

        # print('===== load initial valid data =====')
        q_valid = self._valid_data[:, 0]; p_valid = self._valid_data[:, 1]

        # print('===== label valid data =====')
        q_valid_label = self._valid_label[0]; p_valid_label = self._valid_label[1]
        q_valid_label = q_valid_label[-1]; p_valid_label = p_valid_label[-1]  # only take the last from the list

        assert q_valid.shape == q_valid_label.shape
        # to prepare data at large time step, need to change tau and iterations
        # tau = large time step 0.1 and 1 step
        # print('===== state at large time step 0.1 =====')

        self._state['nsamples_cur'] = self._state['nsamples_ML']
        self._state['tau_cur'] = self._state['tau_long']  # tau = 0.1
        self._state['MD_iterations'] = int(self._state['tau_long']/self._state['tau_cur'])

        pairwise_hnn = self.pair_wise_HNN(self.noML_hamiltonian, self._MLP, **self._state)
        pairwise_hnn.train()
        criterion = self._loss

        n_train_batches = q_train.shape[0] # nsamples
        n_valid_batches = q_valid.shape[0]
        text = ''

        for e in range(1, self._state['nepochs'] + 1 ):

            train_loss = 0.
            valid_loss = 0.
            start = time.time()

            for i in range(n_train_batches):

                q_train_batch, p_train_batch = q_train[i], p_train[i]
                q_train_batch = torch.unsqueeze(q_train_batch, dim=0).to(self._state['_device'])
                p_train_batch = torch.unsqueeze(p_train_batch, dim=0).to(self._state['_device'])

                q_train_label_batch, p_train_label_batch = q_train_label[i], p_train_label[i]
                q_train_label_batch = torch.unsqueeze(q_train_label_batch, dim=0).to(self._state['_device'])
                p_train_label_batch = torch.unsqueeze(p_train_label_batch, dim=0).to(self._state['_device'])
                # print('train label', q_train_label_batch, p_train_label_batch)

                train_label = (q_train_label_batch, p_train_label_batch)

                self._state['phase_space'].set_q(q_train_batch)
                self._state['phase_space'].set_p(p_train_batch)

                # print('======= train combination of MD and ML =======')
                q_train_pred, p_train_pred = self.linear_integrator(**self._state).integrate(pairwise_hnn)
                q_train_pred = q_train_pred.to(self._state['_device']); p_train_pred = p_train_pred.to(self._state['_device'])

                train_predict = (q_train_pred[-1], p_train_pred[-1])
                # print('train pred', q_train_pred[-1], p_train_pred[-1])

                loss1 = criterion(train_predict, train_label)

                self._opt.zero_grad()  # defore the backward pass, use the optimizer object to zero all of the gradients for the variables
                loss1.backward()  # backward pass : compute gradient of the loss wrt models parameters

                self._opt.step()

                train_loss += loss1.item()  # get the scalar output
                # print('loss each batch',loss1.item())

            # eval model
            pairwise_hnn.eval()

            with torch.no_grad():

                for j in range(n_valid_batches):

                    q_valid_batch, p_valid_batch = q_valid[j], p_valid[j]

                    q_valid_batch = torch.unsqueeze(q_valid_batch, dim=0).to(self._state['_device'])
                    p_valid_batch = torch.unsqueeze(p_valid_batch, dim=0).to(self._state['_device'])

                    q_valid_label_batch, p_valid_label_batch = q_valid_label[j], p_valid_label[j]

                    q_valid_label_batch = torch.unsqueeze(q_valid_label_batch, dim=0).to(self._state['_device'])
                    p_valid_label_batch = torch.unsqueeze(p_valid_label_batch, dim=0).to(self._state['_device'])
                    # print('valid label', q_valid_label_batch, p_valid_label_batch)

                    valid_label = (q_valid_label_batch, p_valid_label_batch)

                    self._state['phase_space'].set_q(q_valid_batch)
                    self._state['phase_space'].set_p(p_valid_batch)

                    # print('======= train combination of MD and ML =======')
                    q_valid_pred, p_valid_pred = self.linear_integrator(**self._state).integrate(pairwise_hnn)
                    q_valid_pred = q_valid_pred.to(self._state['_device']); p_valid_pred = p_valid_pred.to(self._state['_device'])

                    valid_predict = (q_valid_pred[-1], p_valid_pred[-1])
                    # print('valid pred', q_valid_pred[-1], q_valid_pred[-1])

                    val_loss1 = criterion(valid_predict, valid_label)

                    valid_loss += val_loss1.item()  # get the scalar output

            train_loss_avg = train_loss / n_train_batches
            valid_loss_avg = valid_loss / n_valid_batches

            end = time.time()

            print('{} epoch:'.format(e), 'train_loss:', train_loss_avg, 'valid_loss:', valid_loss_avg, ' time:', end-start)

            if not os.path.exists('./saved_model/'):
                os.makedirs('./saved_model/')

            self.record_best(valid_loss_avg, './saved_model/nsamples{}_nparticle{}_tau{}_lr{}_h{}_checkpoint.pth'.format(self._state['nsamples_label'],self._state['nparticle'], self._state['tau_cur'],
                                                 self._opt.param_groups[0]['lr'], self._state['n_hidden']))

            text = text + str(e) + ' ' + str(train_loss_avg) + ' ' + str(valid_loss_avg) + '\n'
            with open('nsamples{}_nparticle{}_tau{}_loss.txt'.format(self._state['nsamples_label'],self._state['nparticle'],self._state['tau_cur']), 'w') as fp:
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
            shutil.copyfile(filename, './saved_model/nsamples{}_nparticle{}_tau{}_lr{}_h{}_checkpoint_best.pth'.format(self._state['nsamples_label'],self._state['nparticle'], self._state['tau_cur'],
                                                     self._opt.param_groups[0]['lr'], self._state['n_hidden']))


    def pred_qnp(self, filename):

        # load the models checkpoint
        checkpoint = torch.load('./saved_model/nsamples{}_nparticle{}_tau{}_lr{}_h{}_checkpoint.pth'.format(
            self._state['nsamples_label'],self._state['nparticle'], self._state['tau_long'], self._opt.param_groups[0]['lr'], self._state['n_hidden']))[0]
        print(checkpoint)
        quit()
        # load models weights state_dict
        self._MLP.load_state_dict(checkpoint['model_state_dict'])
        print('Previously trained models weights state_dict loaded...')
        self._opt.load_state_dict(checkpoint['optimizer'])
        print('Previously trained optimizer state_dict loaded...')
        # print("Optimizer's state_dict:")
        # for var_name in self._opt.state_dict():
        #     print(var_name, "\t", self._opt.state_dict()[var_name])

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
