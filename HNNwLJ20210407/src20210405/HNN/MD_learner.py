from HNN.loss import qp_MSE_loss
from parameters.MD_parameters import MD_parameters
from parameters.ML_parameters import ML_parameters
import time

import torch

class MD_learner:

    ''' MD_learner class to help train, validate, retrain, and save '''

    _obj_count = 0

    def __init__(self, linear_integrator_obj, any_HNN_obj, phase_space, opt, data_loader, chk_pt):

        '''
        Parameters
        ----------
        linear_integrator_obj : use for integrator using large time step
        any_HNN_obj : pass any HNN object to this container
        loader : DataLoaders on Custom Datasets
                 two tensors contain train and valid data
                 each shape is [2, niter, nsamples, nparticle, DIM] , here 2 is (q,p)
                 niter is initial and append strike iter so that 2
        '''

        MD_learner._obj_count += 1
        assert (MD_learner._obj_count == 1), type(self).__name__ + " has more than one object"

        self.linear_integrator = linear_integrator_obj
        self.any_HNN = any_HNN_obj
        self.any_network =  any_HNN_obj.net1
        self.data_loader = data_loader
        self._phase_space = phase_space
        self.chk_pt = chk_pt
        self._opt = opt

        self._loss = qp_MSE_loss
        self._current_epoch = 1

    def train_epoch(self):

        self.any_HNN.train()
        train_loss = 0.

        criterion = self._loss

        for step, (input, label) in enumerate(self.data_loader.train_loader):

            self._opt.zero_grad()
            # defore the backward pass, use the optimizer object to zero all of the gradients for the variables

            # input shape, [nsamples, (q,p), nparticle, DIM]
            self._phase_space.set_q(input[:,0,:,:])
            self._phase_space.set_p(input[:,1,:,:])
            qp_list = self.linear_integrator.one_step(self.any_HNN, self._phase_space, self._tau_cur)

            q_predict = qp_list[:,0,:,:]; p_predict = qp_list[:,1,:,:]
            q_label   = label[:,0,:,:]; p_label   = label[:,1,:,:]

            train_predict = (q_predict, p_predict)

            train_label = (q_label, p_label)

            loss1 = criterion(train_predict, train_label)

            loss1.backward()  # backward pass : compute gradient of the loss wrt models parameters
            self._opt.step()

            train_loss += loss1.item() / ML_parameters.batch_size  # get the scalar output

        return train_loss


    def valid_epoch(self):

        self.any_HNN.eval()
        valid_loss = 0.

        criterion = self._loss

        for step, (input, label) in enumerate(self.data_loader.valid_loader):

            self._opt.zero_grad()
            # defore the backward pass, use the optimizer object to zero all of the gradients for the variables

            # input shape, [nsamples, (q,p), nparticle, DIM]
            self._phase_space.set_q(input[:, 0, :, :])
            self._phase_space.set_p(input[:, 1, :, :])
            qp_list = self.linear_integrator.one_step(self.any_HNN, self._phase_space, self._tau_cur)

            q_predict = qp_list[:, 0, :, :]; p_predict = qp_list[:, 1, :, :]
            q_label = label[:, 0, :, :]; p_label = label[:, 1, :, :]

            valid_predict = (q_predict, p_predict)

            valid_label = (q_label, p_label)

            loss1 = criterion(valid_predict, valid_label)

            loss1.backward()  # backward pass : compute gradient of the loss wrt models parameters
            self._opt.step()

            valid_loss += loss1.item() / ML_parameters.batch_size  # get the scalar output

        return valid_loss

    def nepoch(self, save_filename, write_loss_filename):

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

        Returns
        -------
        float
            train loss, valid loss per epoch
        '''

        self._tau_cur = MD_parameters.tau_long

        text = ''

        for e in range(1, ML_parameters.nepoch + 1 ):
            start_epoch = time.time()

            train_loss = self.train_epoch()

            # with torch.no_grad():
            #
            #     valid_loss = self.valid_epoch()

            end_epoch = time.time()

            train_loss_avg = train_loss / (len(self.data_loader.train_loader) // ML_parameters.batch_size)
            # valid_loss_avg = valid_loss / (len(self.data_loader.valid_loader)  // ML_parameters.batch_size)

            self.chk_pt.save_checkpoint(save_filename)

            # print('================ loss each train valid epoch ================')
            print('{} epoch:'.format(e), 'train_loss:', train_loss_avg, ' each epoch time:', end_epoch - start_epoch)

            text = text + str(e) + ' ' + str(train_loss_avg)  + ' ' + str(end_epoch - start_epoch) + '\n'

            self.save_loss_curve(text, write_loss_filename)

            self._current_epoch += 1

    def save_loss_curve(self, text, write_loss_filename):

        with open(write_loss_filename, 'w') as fp:
            fp.write(text)
        fp.close()