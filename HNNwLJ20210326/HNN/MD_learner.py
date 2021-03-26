from .data_io import data_io
from .loss import qp_MSE_loss
from MD_parameters import MD_parameters
from ML_parameters import ML_parameters
from ._checkpoint import _checkpoint
import torch
import shutil
import time
import os
import sys

class MD_learner:

    ''' MD_learner class to help train, validate, retrain, and save '''

    _obj_count = 0

    def __init__(self, linear_integrator_obj, any_HNN_obj, phase_space, init_path, load_path, crash_filename=None):

        '''
        Parameters
        ----------
        linear_integrator_obj : use for integrator using large time step
        any_HNN_obj : pass any HNN object to this container
        init_path : string
                folder name
        load_path : string
                filename to save checkpoint
                default is None
        crash_filename : str, optional
                default is None

        '''

        MD_learner._obj_count += 1
        assert (MD_learner._obj_count == 1), type(self).__name__ + " has more than one object"

        self.linear_integrator = linear_integrator_obj
        self.any_HNN = any_HNN_obj
        self.any_network =  any_HNN_obj.network
        self.noML_hamiltonian = super(type(any_HNN_obj), any_HNN_obj)

        print('-- hi terms -- ',self.noML_hamiltonian.hi())

        self._phase_space = phase_space
        self._data_io_obj = data_io(init_path)

        print("============ train/valid data ===============")
        self.q_train, self.p_train = self._data_io_obj.qp_dataset('train', crash_filename)
        self.q_valid, self.p_valid = self._data_io_obj.qp_dataset('valid')

        print('n. of train data reshape ', self.q_train.shape, self.p_train.shape)
        print('n. of valid data reshape ', self.q_valid.shape, self.p_valid.shape)

        print("============ label for train/valid ===============")
        # qnp x iterations x nsamples x  nparticle x DIM
        self.q_train_label, self.p_train_label = self._data_io_obj.phase_space2label(self.q_train, self.p_train, self.linear_integrator, self._phase_space, self.noML_hamiltonian)
        self.q_valid_label, self.p_valid_label = self._data_io_obj.phase_space2label(self.q_valid, self.p_valid, self.linear_integrator, self._phase_space, self.noML_hamiltonian)

        print('n. of train label reshape ', self.q_train_label.shape, self.p_train_label.shape)
        print('n. of valid label reshape ', self.q_valid_label.shape, self.p_valid_label.shape)

        self._device = ML_parameters.device

        self._opt = ML_parameters.opt.create(self.any_network.parameters())
        print(ML_parameters.opt.name())

        # Assuming optimizer uses lr = 0.001 for all groups
        # lr = 0.001      if epoch < 10
        # lr = 0.0001     if 10 <= epoch < 20
        # lr = 0.00001    if 20 <= epoch < 30

        # # gamma = decaying factor
        # self._scheduler = StepLR(self._opt, step_size=2, gamma=0.99)

        self._loss = qp_MSE_loss
        self._current_epoch = 1

        self._checkpoint = _checkpoint(self.any_network, self._opt, self._current_epoch)
        self._checkpoint.load_checkpoint(load_path)


    def train_valid_epoch(self, save_path, best_model_path, loss_curve):

        ''' function to train and valid each epoch

        parameters
        -------
        save_path : dir + filename
        loss_curve : filename
        best_model_path : dir + filename
        nsamples_cur : int
                1 : load one sample in pair-wise HNN
                batch : load batch in field HNN
        tau_cur : float
                long time step
        MD_iterations : int
                1 : integration of large time step
        random_ordered_train_nsamples : int
                n. of samples for train
        random_ordered_valid_nsamples : int
                n. of samples for valid

        Returns
        -------
        float
            train loss, valid loss per epoch
        '''

        nsamples_cur =  MD_parameters.nsamples_ML
        self._tau_cur = MD_parameters.tau_long
        MD_iterations = int( MD_parameters.tau_long / self._tau_cur )

        random_ordered_train_nsamples = self.q_train.shape[0] # nsamples
        random_ordered_valid_nsamples = self.q_valid.shape[0] # nsamples

        print('prepare train nsamples_cur, tau_cur, MD_iterations')
        print(nsamples_cur, self._tau_cur, MD_iterations)
        print('nsample', random_ordered_train_nsamples, random_ordered_valid_nsamples)

        self.any_HNN.train()
        criterion = self._loss

        text = ''

        for e in range(1, ML_parameters.nepoch + 1 ):

            train_loss = 0.
            valid_loss = 0.

            # Decay Learning Rate
            # curr_lr = self._scheduler.get_lr()
            # self._scheduler.step()

            # print('epoch', e, 'lr', curr_lr)
            start_epoch = time.time()

            for i in range(0, random_ordered_train_nsamples, nsamples_cur):

                q_train_batch, p_train_batch = self.q_train[i:i+nsamples_cur], self.p_train[i:i+nsamples_cur] # each sample
                q_train_label_batch, p_train_label_batch = self.q_train_label[:,i:i+nsamples_cur], self.p_train_label[:,i:i+nsamples_cur]

                self._phase_space.set_q(q_train_batch.to(self._device))
                self._phase_space.set_p(p_train_batch.to(self._device))

                # print('======= train combination of MD and ML =======')
                q_train_pred, p_train_pred = self.linear_integrator.step( self.any_HNN, self._phase_space, MD_iterations, self._tau_cur)

                train_predict = (q_train_pred, p_train_pred)
                train_label = (q_train_label_batch.to(self._device), p_train_label_batch.to(self._device))

                loss1 = criterion(train_predict, train_label)

                self._opt.zero_grad()  # defore the backward pass, use the optimizer object to zero all of the gradients for the variables
                loss1.backward()  # backward pass : compute gradient of the loss wrt models parameters
                self._opt.step()

                train_loss += loss1.item() / nsamples_cur  # get the scalar output
                quit()
            # eval model

            self.any_HNN.eval()

            with torch.no_grad():

                for j in range(0, random_ordered_valid_nsamples, nsamples_cur):

                    q_valid_batch, p_valid_batch = self.q_valid[j:j+nsamples_cur], self.p_valid[j:j+nsamples_cur]
                    q_valid_label_batch, p_valid_label_batch = self.q_valid_label[:,j:j+nsamples_cur], self.p_valid_label[:,j:j+nsamples_cur]

                    self._phase_space.set_q(q_valid_batch.to(self._device))
                    self._phase_space.set_p(p_valid_batch.to(self._device))

                    # print('======= valid combination of MD and ML =======')
                    q_valid_pred, p_valid_pred = self.linear_integrator.step( self.any_HNN, self._phase_space, MD_iterations, self._tau_cur)

                    valid_predict = (q_valid_pred, p_valid_pred)
                    valid_label = (q_valid_label_batch.to(self._device), p_valid_label_batch.to(self._device))

                    val_loss1 = criterion(valid_predict, valid_label)

                    valid_loss += val_loss1.item() / nsamples_cur # get the scalar output


            end_epoch = time.time()

            train_loss_avg = train_loss / (random_ordered_train_nsamples // nsamples_cur)
            valid_loss_avg = valid_loss / (random_ordered_valid_nsamples  // nsamples_cur)

            # # Decay Learning Rate after every epoch
            # curr_lr = self._scheduler.get_lr()
            # self._scheduler.step()

            # print('================ loss each train valid epoch ================')
            print('{} epoch:'.format(e), 'train_loss:', train_loss_avg, 'valid_loss:', valid_loss_avg, ' each epoch time:', end_epoch - start_epoch)

            self._checkpoint.save_checkpoint(valid_loss_avg, save_path, best_model_path, self._current_epoch)

            text = text + str(e) + ' ' + str(train_loss_avg) + ' ' + str(valid_loss_avg) + ' ' + str(end_epoch - start_epoch) + '\n'
            with open(loss_curve, 'w') as fp:
                fp.write(text)
            fp.close()

            self._current_epoch += 1
