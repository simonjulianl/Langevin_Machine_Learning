from HNN.loss                 import qp_MSE_loss
from HNN.checkpoint           import checkpoint
import time

import torch

class MD_learner:

    ''' MD_learner class to help train, validate, retrain, and save '''

    _obj_count = 0

    def __init__(self, linear_integrator_obj, 
                 any_HNN_obj, phase_space, 
                 opt, data_loader, chk_pt_file=None):

        '''
        Parameters
        ----------
        linear_integrator_obj : use for integrator using large time step
        any_HNN_obj : pass any HNN object to this container
        phase_space : contains q_list, p_list as input
                    q list shape is [nsamples, nparticle, DIM]
        opt         : create one optimizer from two models parameters
        data_loader : DataLoaders on Custom Datasets
                 two tensors contain train and valid data
                 each shape is [nsamples, 2, niter, nparticle, DIM] , here 2 is (q,p)
                 niter is initial and append strike iter so that 2
        chk_pt_file : checkpoint file for save or load them
                 default is None
        '''

        MD_learner._obj_count += 1
        assert (MD_learner._obj_count == 1), type(self).__name__ + " has more than one object"

        self.linear_integrator = linear_integrator_obj
        self.any_HNN = any_HNN_obj
        self.data_loader = data_loader

        self._phase_space = phase_space
        self._opt = opt
        self.chk_pt = checkpoint(self.any_HNN.get_netlist(), self._opt)

        if chk_pt_file is not None: self.chk_pt.load_checkpoint(chk_pt_file)

        self._loss      = qp_MSE_loss
        self.batch_size = self.data_loader.batch_size

        self.tau_cur    = self.data_loader.data_set.train_set.data_tau_long
        self.boxsize    = self.data_loader.data_set.train_set.data_boxsize

        self._phase_space.set_boxsize(self.boxsize)

    # ===================================================
    def train_one_epoch(self):
        ''' function to train one epoch'''

        self.any_HNN.train()
        train_loss = 0.

        criterion = self._loss

        for step, (input, label) in enumerate(self.data_loader.train_loader):

            self._opt.zero_grad()
            # before the backward pass, use the optimizer object to zero all of the gradients for the variables

            # input shape, [nsamples, (q,p)=2, nparticle, DIM]
            self._phase_space.set_q(input[:, 0, :, :])
            self._phase_space.set_p(input[:, 1, :, :])

            qp_list = self.linear_integrator.one_step(self.any_HNN, self._phase_space, self.tau_cur)
            # qp_list shape, [nsamples, (q,p)=2, nparticle, DIM]

            q_predict = qp_list[:, 0, :, :]; p_predict = qp_list[:, 1, :, :]
            # q_predict shape, [nsamples, nparticle, DIM]

            q_label   = label[:, 0, :, :]  ; p_label   = label[:, 1, :, :]
            # q_label shape, [nsamples, nparticle, DIM]

            train_predict = (q_predict, p_predict)
            train_label = (q_label, p_label)

            loss = criterion(train_predict, train_label)

            loss.backward()  # backward pass : compute gradient of the loss wrt models parameters
            self._opt.step()

            train_loss += loss.item() / self.batch_size  # get the scalar output

        return train_loss

    # ===================================================
    def valid_one_epoch(self):
        ''' function to validate one epoch'''

        self.any_HNN.eval()
        valid_loss = 0.

        criterion = self._loss

        for step, (input, label) in enumerate(self.data_loader.valid_loader):

            self._opt.zero_grad()
            # defore the backward pass, use the optimizer object to zero all of the gradients for the variables

            # input shape, [nsamples, (q,p)=2, nparticle, DIM]
            self._phase_space.set_q(input[:, 0, :, :])
            self._phase_space.set_p(input[:, 1, :, :])
            qp_list = self.linear_integrator.one_step(self.any_HNN, self._phase_space, self.tau_cur)
            # qp_list shape, [nsamples, (q,p)=2, nparticle, DIM]

            q_predict = qp_list[:, 0, :, :]; p_predict = qp_list[:, 1, :, :]
            # q_predict shape, [nsamples, nparticle, DIM]

            q_label   = label[:, 0, :, :]  ; p_label   = label[:, 1, :, :]
            # q_label shape, [nsamples, nparticle, DIM]

            valid_predict = (q_predict, p_predict)
            valid_label = (q_label, p_label)

            loss = criterion(valid_predict, valid_label)

            loss.backward()  # backward pass : compute gradient of the loss wrt models parameters
            self._opt.step()

            valid_loss += loss.item() / self.batch_size  # get the scalar output

        return valid_loss

    # ===================================================
    def nepoch(self, nepoch, write_chk_pt_filename, write_loss_filename):

        ''' function to train and valid more than one epoch

        parameters
        -------
        write_chk_pt_filename : filename to save checkpoints
        write_loss_filename   : path + filename to save loss values

        Returns
        -------
        float
            train loss, valid loss every epoch
        '''

        text = ''

        train_set_len = self.data_loader.data_set.train_set.qp_list_input.shape[0]
        train_no_batch_size = train_set_len // self.batch_size

        # valid_set_len = self.data_loader.data_set.valid_set.qp_list_input.shape[0]
        # valid_no_batch_size = valid_set_len // self.batch_size

        for e in range(1, nepoch + 1 ):
            print('e', e)
            start_epoch = time.time()
            train_loss = self.train_one_epoch()

            # with torch.no_grad():
            #     valid_loss = self.valid_one_epoch()
            end_epoch = time.time()

            train_loss_avg = train_loss / train_no_batch_size
            # valid_loss_avg = valid_loss / valid_no_batch_size

            self.chk_pt.save_checkpoint(write_chk_pt_filename)

            # print('================ loss each train valid epoch ================')
            print('{} epoch:'.format(e), 'train_loss:', train_loss_avg, ' each epoch time:', end_epoch - start_epoch)

            text = text + str(e) + ' ' + str(train_loss_avg)  + ' ' + str(end_epoch - start_epoch) + '\n'

            self.save_loss_curve(text, write_loss_filename)

    # ===================================================
    def save_loss_curve(self, text, write_loss_filename):
        ''' function to save the loss every epoch '''

        with open(write_loss_filename, 'w') as fp:
            fp.write(text)
        fp.close()
