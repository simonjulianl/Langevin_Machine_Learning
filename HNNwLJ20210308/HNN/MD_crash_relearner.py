from .data_io import data_io
from .loss import qp_MSE_loss
import torch.optim as optim
from MD_parameters import MD_parameters
from ML_parameters import ML_parameters
# from .models import pair_wise_zero
from torch.optim.lr_scheduler import StepLR
import torch
import shutil
import time
import os
import sys


class MD_crash_relearner:

    _obj_count = 0

    def __init__(self, linear_integrator_obj, any_HNN_obj, phase_space, path):

        MD_crash_relearner._obj_count += 1
        assert (MD_crash_relearner._obj_count == 1), type(self).__name__ + " has more than one object"

        self.linear_integrator = linear_integrator_obj
        self.any_HNN = any_HNN_obj
        self.any_network =  any_HNN_obj.network
        self.noML_hamiltonian = super(type(any_HNN_obj), any_HNN_obj)

        print('-- hi terms -- ',self.noML_hamiltonian.hi())
        # terms = self.noML_hamiltonian.get_terms()

        self._phase_space = phase_space
        self._data_io_obj = data_io(path)

        print("============ start data loaded ===============")
        start_data_load = time.time()

        # _crash_data = self._data_io_obj.loadq_p('test_before_crash')
        # _train_data = self._data_io_obj.loadq_p('train')

        _train_data = self._data_io_obj.hamiltonian_balance_dataset('test_before_crash', 'train')
        self.train_data = self._data_io_obj.hamiltonian_dataset(_train_data)
        self.train_data = self.train_data[:4]
        print('n. of data', self.train_data.shape)

        _valid_data = self._data_io_obj.loadq_p('valid')
        self.valid_data = self._data_io_obj.hamiltonian_dataset(_valid_data)
        self.train_data = self.train_data[:2]
        print('n. of data', self.valid_data.shape)

        end_data_load = time.time()

        print('data loaded time :', end_data_load - start_data_load)
        print("============= end data loaded ================")

        print("============ start data label ===============")
        # qnp x iterations x nsamples x  nparticle x DIM

        start_train_label = time.time()
        self.train_label = self._data_io_obj.phase_space2label(self.train_data, self.linear_integrator, self._phase_space, self.noML_hamiltonian)
        end_train_label = time.time()
        print('prepare train_label time:', end_train_label - start_train_label)

        start_valid_label = time.time()
        self.valid_label = self._data_io_obj.phase_space2label(self.valid_data, self.linear_integrator, self._phase_space, self.noML_hamiltonian)
        end_valid_label = time.time()
        print('prepare valid_label time:', end_valid_label - start_valid_label)
        print("============= end data label =================")

        # print('===== load initial train data =====')
        self._q_train = self.train_data[:,0]; self._p_train = self.train_data[:,1]
        print('n. of train data reshape ', self._q_train.shape, self._p_train.shape)

        # print('===== label train data =====')
        self.q_train_label = self.train_label[0]; self.p_train_label = self.train_label[1]
        self._q_train_label = self.q_train_label[-1]; self._p_train_label = self.p_train_label[-1] # only take the last from the list
        print('n. of train label reshape ', self._q_train_label.shape, self._p_train_label.shape)

        assert self._q_train.shape == self._q_train_label.shape
        assert self._p_train.shape == self._p_train_label.shape

        # print('===== load initial valid data =====')
        self._q_valid = self.valid_data[:, 0]; self._p_valid = self.valid_data[:, 1]
        print('n. of valid data reshape ', self._q_valid.shape, self._p_valid.shape)

        # print('===== label valid data =====')
        self.q_valid_label = self.valid_label[0]; self.p_valid_label = self.valid_label[1]
        self._q_valid_label = self.q_valid_label[-1]; self._p_valid_label = self.p_valid_label[-1]  # only take the last from the list
        print('n. of valid label reshape ', self._q_valid_label.shape, self._p_valid_label.shape)

        assert self._q_valid.shape == self._q_valid_label.shape
        assert self._p_valid.shape == self._p_valid_label.shape

        self._device = ML_parameters.device

        if ML_parameters.optimizer == 'Adam':
            self._opt = optim.Adam(self.any_network.parameters(), lr = ML_parameters.lr)

        elif ML_parameters.optimizer == 'SGD':
            self._opt = optim.SGD(self.any_network.parameters(), lr=ML_parameters.lr)

        else:
            sys.exit(1)

        print(type(self._opt).__name__)


        # Assuming optimizer uses lr = 0.001 for all groups
        # lr = 0.001      if epoch < 10
        # lr = 0.0001     if 10 <= epoch < 20
        # lr = 0.00001    if 20 <= epoch < 30

        # # gamma = decaying factor
        # self._scheduler = StepLR(self._opt, step_size=2, gamma=0.99)

        self._loss = qp_MSE_loss
        self._current_epoch = 1
        # initialize best models
        self._best_validation_loss = float('inf')

    def load_checkpoint(self, load_path):

        if os.path.isfile(load_path):
            print("=> loading checkpoint '{}'".format(load_path))
            checkpoint = torch.load(load_path)[0]
            # print(checkpoint)
            # load models weights state_dict
            self.any_network.load_state_dict(checkpoint['model_state_dict'])
            print('Previously trained models weights state_dict loaded...')
            self._opt.load_state_dict(checkpoint['optimizer'])
            print('Previously trained optimizer state_dict loaded...')
            # self._scheduler = checkpoint['scheduler']
            # print('Previously trained scheduler state_dict loaded...')
            self._current_epoch = checkpoint['epoch'] + 1
            print('Previously trained epoch state_dict loaded...')
            print('current_epoch', self._current_epoch)

            if not os.path.exists('./retrain_saved_model/'):
                os.makedirs('./retrain_saved_model/')

        else:
            print("=> no checkpoint found at '{}'".format(load_path))

            if not os.path.exists('./saved_model/'):
                os.makedirs('./saved_model/')
            # # epoch, best_precision, loss_train
            # return 1, 0, []

    def save_checkpoint(self, validation_loss, save_path, best_model_path):

        is_best = validation_loss < self._best_validation_loss
        self._best_validation_loss = min(validation_loss, self._best_validation_loss)

        torch.save(({
                'epoch': self._current_epoch,
                'model_state_dict' : self.any_network.state_dict(),
                'best_validation_loss' : self._best_validation_loss,
                'optimizer': self._opt.state_dict(),
                # 'scheduler' : self._scheduler
                }, is_best), save_path)

        if is_best:
            shutil.copyfile(save_path, best_model_path)

    # phase_space consist of minibatch data
    def train_valid_epoch(self, save_path, best_model_path, loss_curve):

        # to prepare data at large time step, need to change tau and iterations
        # tau = large time step 0.1 and 1 step
        # print('===== state at large time step 0.1 =====')
        nsamples_cur =  MD_parameters.nsamples_ML
        self._tau_cur = MD_parameters.tau_long # tau = 0.1
        MD_iterations = int( MD_parameters.tau_long / self._tau_cur )

        print('prepare train nsamples_cur, tau_cur, MD_iterations')
        print(nsamples_cur, self._tau_cur, MD_iterations)

        random_ordered_train_nsamples = self._q_train.shape[0] # nsamples
        random_ordered_valid_nsamples = self._q_valid.shape[0] # nsamples
        print('nsample', random_ordered_train_nsamples, random_ordered_valid_nsamples)
        # n_train_batch = MD_parameters.nparticle * (MD_parameters.nparticle - 1)
        # n_valid_batch

        # any_hnn = self.any_HNN
        self.any_HNN.train()
        criterion = self._loss

        text = ''

        start = time.time()

        for e in range(1, ML_parameters.nepoch + 1 ):

            train_loss = 0.
            valid_loss = 0.

            # Decay Learning Rate
            # curr_lr = self._scheduler.get_lr()
            # self._scheduler.step()

            # print('epoch', e, 'lr', curr_lr)
            start_epoch = time.time()
            start_epoch_train = time.time()

            for i in range(random_ordered_train_nsamples): # load each sample for loop
                # print(i)
                start_batch_train = time.time()

                q_train_batch, p_train_batch = self._q_train[i], self._p_train[i] # each sample
                q_train_batch = torch.unsqueeze(q_train_batch, dim=0).to(self._device)
                p_train_batch = torch.unsqueeze(p_train_batch, dim=0).to(self._device)

                q_train_label_batch, p_train_label_batch = self._q_train_label[i], self._p_train_label[i]
                q_train_label_batch = torch.unsqueeze(q_train_label_batch, dim=0).to(self._device)
                p_train_label_batch = torch.unsqueeze(p_train_label_batch, dim=0).to(self._device)
                # print('train label', q_train_label_batch, p_train_label_batch)

                train_label = (q_train_label_batch, p_train_label_batch)


                self._phase_space.set_q(q_train_batch)
                self._phase_space.set_p(p_train_batch)

                # print('======= train combination of MD and ML =======')
                start_pred = time.time()

                q_train_pred, p_train_pred = self.linear_integrator.step( self.any_HNN, self._phase_space, MD_iterations, nsamples_cur, self._tau_cur)
                # q_train_pred = torch.zeros(torch.unsqueeze(q_train_label_batch, dim=0).shape,requires_grad=True)
                # p_train_pred = torch.zeros(torch.unsqueeze(q_train_label_batch, dim=0).shape,requires_grad=True)
                end_pred = time.time()

                q_train_pred = q_train_pred.to(self._device); p_train_pred = p_train_pred.to(self._device)

                train_predict = (q_train_pred[-1], p_train_pred[-1])
                # print('train pred', q_train_pred[-1], p_train_pred[-1])

                start_loss = time.time()
                loss1 = criterion(train_predict, train_label)
                end_loss = time.time()

                start_backward = time.time()
                self._opt.zero_grad()  # defore the backward pass, use the optimizer object to zero all of the gradients for the variables
                loss1.backward()  # backward pass : compute gradient of the loss wrt models parameters
                end_backward = time.time()

                self._opt.step()

                train_loss += loss1.item()  # get the scalar output

            end_epoch_train = time.time()

            # print('============================================================')
            # print('loss each train epoch time', end_epoch_train - start_epoch_train)
            # print('============================================================')


            # eval model

            self.any_HNN.eval()

            with torch.no_grad():

                start_epoch_valid = time.time()

                for j in range(random_ordered_valid_nsamples):

                    start_batch_valid = time.time()

                    q_valid_batch, p_valid_batch = self._q_valid[j], self._p_valid[j]

                    q_valid_batch = torch.unsqueeze(q_valid_batch, dim=0).to(self._device)
                    p_valid_batch = torch.unsqueeze(p_valid_batch, dim=0).to(self._device)

                    q_valid_label_batch, p_valid_label_batch = self._q_valid_label[j], self._p_valid_label[j]

                    q_valid_label_batch = torch.unsqueeze(q_valid_label_batch, dim=0).to(self._device)
                    p_valid_label_batch = torch.unsqueeze(p_valid_label_batch, dim=0).to(self._device)
                    # print('valid label', q_valid_label_batch, p_valid_label_batch)

                    valid_label = (q_valid_label_batch, p_valid_label_batch)

                    self._phase_space.set_q(q_valid_batch)
                    self._phase_space.set_p(p_valid_batch)

                    # print('======= valid combination of MD and ML =======')
                    q_valid_pred, p_valid_pred = self.linear_integrator.step( self.any_HNN, self._phase_space, MD_iterations, nsamples_cur, self._tau_cur)
                    q_valid_pred = q_valid_pred.to(self._device); p_valid_pred = p_valid_pred.to(self._device)

                    valid_predict = (q_valid_pred[-1], p_valid_pred[-1])
                    # print('valid pred', q_valid_pred[-1], q_valid_pred[-1])

                    val_loss1 = criterion(valid_predict, valid_label)

                    valid_loss += val_loss1.item()  # get the scalar output

                    end_batch_valid = time.time()
                    # print('loss each valid batch time', end_batch_valid - start_batch_valid)

                end_epoch_valid = time.time()

            # print('loss each valid epoch time', end_epoch_valid - start_epoch_valid)
            # print('============================================================')

            end_epoch = time.time()

            train_loss_avg = train_loss / random_ordered_train_nsamples
            valid_loss_avg = valid_loss / random_ordered_valid_nsamples

            # # Decay Learning Rate after every epoch
            # curr_lr = self._scheduler.get_lr()
            # self._scheduler.step()

            # print('================ loss each train valid epoch ================')
            print('{} epoch:'.format(e), 'train_loss:', train_loss_avg, 'valid_loss:', valid_loss_avg, ' each epoch time:', end_epoch - start_epoch)

            self.save_checkpoint(valid_loss_avg, save_path, best_model_path)

            text = text + str(e) + ' ' + str(train_loss_avg) + ' ' + str(valid_loss_avg) + ' ' + str(end_epoch - start_epoch) + '\n'
            with open(loss_curve, 'w') as fp:
                fp.write(text)
            fp.close()

            self._current_epoch += 1

        end = time.time()

        # print('end training... used parameter: tau long: {}, tau short: {}, epochs time: {}'.format(MD_parameters.tau_long, MD_parameters.tau_short, end - start))


