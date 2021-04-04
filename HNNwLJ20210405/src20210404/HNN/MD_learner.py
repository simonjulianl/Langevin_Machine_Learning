from HNN.loss import qp_MSE_loss
from parameters.MD_parameters import MD_parameters
from parameters.ML_parameters import ML_parameters

import torch

class MD_learner:

    ''' MD_learner class to help train, validate, retrain, and save '''

    _obj_count = 0

    def __init__(self, linear_integrator_obj, any_HNN_obj, phase_space, opt, dataset, chk_pt):

        '''
        Parameters
        ----------
        linear_integrator_obj : use for integrator using large time step
        any_HNN_obj : pass any HNN object to this container
        dataset : tuple
                 two tensors contain train and valid data
                 each shape is [2, niter, nsamples, nparticle, DIM] , here 2 is (q,p)
                 niter is initial and append strike iter so that 2
        '''

        MD_learner._obj_count += 1
        assert (MD_learner._obj_count == 1), type(self).__name__ + " has more than one object"

        self.linear_integrator = linear_integrator_obj
        self.any_HNN = any_HNN_obj
        self.any_network =  any_HNN_obj.network

        self._phase_space = phase_space
        self.chk_pt = chk_pt

        self.q_train = dataset[0][0][0]; self.p_train = dataset[0][0][1]
        self.q_valid = dataset[0][1][0]; self.p_valid = dataset[0][1][1]

        self.q_train_label = dataset[1][0][0]; self.p_train_label = dataset[1][0][1]
        self.q_valid_label = dataset[1][1][0]; self.p_valid_label = dataset[1][1][1]

        self.no_train_nsamples = self.q_train.shape[0] # nsamples
        self.no_valid_nsamples = self.q_valid.shape[0] # nsamples

        self._device = ML_parameters.device

        self._opt = opt

        self._loss = qp_MSE_loss
        self._current_epoch = 1

    def train_epoch(self,  niter, nsamples_cur):

        self.any_HNN.train()
        train_loss = 0.

        criterion = self._loss

        for i in range(0, self.no_train_nsamples, nsamples_cur):

            q_train_batch, p_train_batch = self.q_train[i:i + nsamples_cur], self.p_train[i:i + nsamples_cur]  # each sample
            q_train_label_batch, p_train_label_batch = self.q_train_label[:, i:i + nsamples_cur], self.p_train_label[:, i:i + nsamples_cur]

            self._phase_space.set_q(q_train_batch.to(self._device))
            self._phase_space.set_p(p_train_batch.to(self._device))

            # print('======= train combination of MD and ML =======')
            q_train_pred, p_train_pred = self.linear_integrator.step(self.any_HNN, self._phase_space, niter, self._tau_cur)

            train_predict = (q_train_pred, p_train_pred)
            train_label = (q_train_label_batch.to(self._device), p_train_label_batch.to(self._device))

            loss1 = criterion(train_predict, train_label)

            self._opt.zero_grad()  # defore the backward pass, use the optimizer object to zero all of the gradients for the variables
            loss1.backward()  # backward pass : compute gradient of the loss wrt models parameters
            self._opt.step()

            train_loss += loss1.item() / nsamples_cur  # get the scalar output

        return train_loss


    def valid_epoch(self,  niter, nsamples_cur):

        self.any_HNN.eval()
        valid_loss = 0.

        criterion = self._loss

        for j in range(0, self.no_valid_nsamples, nsamples_cur):
            q_valid_batch, p_valid_batch = self.q_valid[j:j + nsamples_cur], self.p_valid[j:j + nsamples_cur]
            q_valid_label_batch, p_valid_label_batch = self.q_valid_label[:, j:j + nsamples_cur], self.p_valid_label[:, j:j + nsamples_cur]

            self._phase_space.set_q(q_valid_batch.to(self._device))
            self._phase_space.set_p(p_valid_batch.to(self._device))

            # print('======= valid combination of MD and ML =======')
            q_valid_pred, p_valid_pred = self.linear_integrator.step(self.any_HNN, self._phase_space, niter,  self._tau_cur)

            valid_predict = (q_valid_pred, p_valid_pred)
            valid_label = (q_valid_label_batch.to(self._device), p_valid_label_batch.to(self._device))

            val_loss1 = criterion(valid_predict, valid_label)

            valid_loss += val_loss1.item() / nsamples_cur  # get the scalar output

        return valid_loss

    def nepoch(self, niter, save_filename, save_loss_curve):

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
        no_train_nsamples : int
                n. of samples for train
        no_valid_nsamples : int
                n. of samples for valid

        Returns
        -------
        float
            train loss, valid loss per epoch
        '''

        nsamples_cur =  MD_parameters.nsamples_ML
        self._tau_cur = MD_parameters.tau_long

        text = ''

        for e in range(1, ML_parameters.nepoch + 1 ):

            train_loss = self.train_epoch(niter, nsamples_cur)

            with torch.no_grad():

                valid_loss = self.valid_epoch(niter, nsamples_cur)

            train_loss_avg = train_loss / (self.no_train_nsamples // nsamples_cur)
            valid_loss_avg = valid_loss / (self.no_valid_nsamples  // nsamples_cur)

            self.chk_pt.save_checkpoint(save_filename)

            # print('================ loss each train valid epoch ================')
            print('{} epoch:'.format(e), 'train_loss:', train_loss_avg, 'valid_loss:', valid_loss_avg, ' each epoch time:', end_epoch - start_epoch)

            text = text + str(e) + ' ' + str(train_loss_avg) + ' ' + str(valid_loss_avg) + ' ' + str(end_epoch - start_epoch) + '\n'

            self.save_loss_curve(text, save_loss_curve)

            self._current_epoch += 1

    def save_loss_curve(self, text, write_loss_filename):

        with open(write_loss_filename, 'w') as fp:
            fp.write(text)
        fp.close()