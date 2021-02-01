from .data_io import data_io
from .loss import qp_MSE_loss
import torch.optim as optim
from HNNwLJ20210128.parameters.MD_paramaters import MD_parameters
from HNNwLJ20210128.parameters.ML_paramaters import ML_parameters
from .models import pair_wise_MLP
# from .models import pair_wise_zero
import torch
import shutil
import time
import os


class MD_learner:

    def __init__(self, linear_integrator_obj, any_HNN_obj, phase_space, filename):

        self.linear_integrator = linear_integrator_obj
        self.any_HNN = any_HNN_obj
        self.any_HNN_class = type(any_HNN_obj)
        self.noML_hamiltonian = super(type(any_HNN_obj), any_HNN_obj)

        print('-- hi terms -- ',self.noML_hamiltonian.hi())
        # terms = self.noML_hamiltonian.get_terms()

        self._phase_space = phase_space

        print("===========start data prepared===============")
        self._data_io_obj = data_io(phase_space, filename)

        # nsamples X qnp X nparticle X  DIM
        self.train_data, self.valid_data = self._data_io_obj.hamiltonian_dataset(ratio = ML_parameters.ratio)

        # qnp x iterations x nsamples x  nparticle x DIM
        print('===========train_label===========')
        self.train_label = self._data_io_obj.phase_space2label(self.train_data, self.linear_integrator, self.noML_hamiltonian)
        print('===========end train_label===========')

        print('===========valid_label===========')
        self.valid_label = self._data_io_obj.phase_space2label(self.valid_data, self.linear_integrator, self.noML_hamiltonian)
        print('===========end valid_label===========')

        # print('===== load initial train data =====')
        self._q_train = self.train_data[:,0]; self._p_train = self.train_data[:,1]

        # print('===== label train data =====')
        self.q_train_label = self.train_label[0]; self.p_train_label = self.train_label[1]
        self._q_train_label = self.q_train_label[-1]; self._p_train_label = self.p_train_label[-1] # only take the last from the list

        assert self._q_train.shape == self._q_train_label.shape
        assert self._p_train.shape == self._p_train_label.shape

        # print('===== load initial valid data =====')
        self._q_valid = self.valid_data[:, 0]; self._p_valid = self.valid_data[:, 1]

        # print('===== label valid data =====')
        self.q_valid_label = self.valid_label[0]; self.p_valid_label = self.valid_label[1]
        self._q_valid_label = self.q_valid_label[-1]; self._p_valid_label = self.p_valid_label[-1]  # only take the last from the list

        assert self._q_valid.shape == self._q_valid_label.shape
        assert self._p_valid.shape == self._p_valid_label.shape

        print("===========end data prepared===============")

        self._device = ML_parameters.device

        self._MLP = pair_wise_MLP().to(self._device)
        self._opt = optim.Adam(self._MLP.parameters(), lr = ML_parameters.lr)
        self._loss = qp_MSE_loss

        self._current_epoch = 1
        # initialize best models
        self._best_validation_loss = float('inf')


    # phase_space consist of minibatch data
    def train_valid_epoch(self):

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

        # n_train_batch = MD_parameters.nparticle * (MD_parameters.nparticle - 1)
        # n_valid_batch

        any_hnn = self.any_HNN_class(self._MLP)
        any_hnn.train()
        criterion = self._loss

        text = ''

        for e in range(1, ML_parameters.nepoch + 1 ):

            train_loss = 0.
            valid_loss = 0.
            start = time.time()

            for i in range(random_ordered_train_nsamples): # load each sample for loop

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
                q_train_pred, p_train_pred = self.linear_integrator.integrate( any_hnn, self._phase_space, MD_iterations, nsamples_cur, self._tau_cur)
                q_train_pred = q_train_pred.to(self._device); p_train_pred = p_train_pred.to(self._device)

                train_predict = (q_train_pred[-1], p_train_pred[-1])
                # print('train pred', q_train_pred[-1], p_train_pred[-1])

                loss1 = criterion(train_predict, train_label)

                self._opt.zero_grad()  # defore the backward pass, use the optimizer object to zero all of the gradients for the variables
                loss1.backward()  # backward pass : compute gradient of the loss wrt models parameters

                self._opt.step()

                train_loss += loss1.item()  # get the scalar output
                # print('loss each batch',loss1.item())

            # eval model

            any_hnn.eval()

            with torch.no_grad():

                for j in range(random_ordered_valid_nsamples):

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
                    q_valid_pred, p_valid_pred = self.linear_integrator.integrate( any_hnn, self._phase_space, MD_iterations, nsamples_cur, self._tau_cur)
                    q_valid_pred = q_valid_pred.to(self._device); p_valid_pred = p_valid_pred.to(self._device)

                    valid_predict = (q_valid_pred[-1], p_valid_pred[-1])
                    # print('valid pred', q_valid_pred[-1], q_valid_pred[-1])

                    val_loss1 = criterion(valid_predict, valid_label)

                    valid_loss += val_loss1.item()  # get the scalar output

            train_loss_avg = train_loss / random_ordered_train_nsamples
            valid_loss_avg = valid_loss / random_ordered_valid_nsamples

            end = time.time()

            print('{} epoch:'.format(e), 'train_loss:', train_loss_avg, 'valid_loss:', valid_loss_avg, ' time:', end-start)

            if not os.path.exists('./saved_model/'):
                os.makedirs('./saved_model/')

            self.record_best(valid_loss_avg, './saved_model/nsamples{}_nparticle{}_tau{}_lr{}_h{}_checkpoint.pth'.format( MD_parameters.nsamples, MD_parameters.nparticle, self._tau_cur,
                                                 self._opt.param_groups[0]['lr'], ML_parameters.MLP_nhidden))

            text = text + str(e) + ' ' + str(train_loss_avg) + ' ' + str(valid_loss_avg) + '\n'
            with open('nsamples{}_nparticle{}_tau{}_loss.txt'.format(MD_parameters.nsamples, MD_parameters.nparticle, self._tau_cur), 'w') as fp:
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
            shutil.copyfile(filename, './saved_model/nsamples{}_nparticle{}_tau{}_lr{}_h{}_checkpoint_best.pth'.format( MD_parameters.nsamples, MD_parameters.nparticle, self._tau_cur,
                                                     self._opt.param_groups[0]['lr'], ML_parameters.MLP_nhidden))


    def pred_qnp(self, filename):

        # load the models checkpoint
        checkpoint = torch.load('./saved_model/nsamples{}_nparticle{}_tau{}_lr{}_h{}_checkpoint.pth'.format(
            MD_parameters.nsamples, MD_parameters.nparticle, MD_parameters.tau_long, self._opt.param_groups[0]['lr'], ML_parameters.MLP_nhidden))[0]
        print(checkpoint)

        # load models weights state_dict
        self._MLP.load_state_dict(checkpoint['model_state_dict'])
        print('Previously trained models weights state_dict loaded...')
        self._opt.load_state_dict(checkpoint['optimizer'])
        print('Previously trained optimizer state_dict loaded...')
        # print("Optimizer's state_dict:")
        # for var_name in self._opt.state_dict():
        #     print(var_name, "\t", self._opt.state_dict()[var_name])

        # initial data
        q_list, p_list = self._phase_space.read(filename, nsamples= MD_parameters.nsamples)
        # print(q_list, p_list)

        self._phase_space.set_q(torch.unsqueeze(q_list[0], dim=0).to(self._device))
        self._phase_space.set_p(torch.unsqueeze(p_list[0], dim=0).to(self._device))

        nsamples_cur = MD_parameters.nsamples_ML
        self._tau_cur = MD_parameters.tau_long  # tau = 0.1
        MD_iterations = int( MD_parameters.tau_long / self._tau_cur)

        any_hnn = self.any_HNN_class(self._MLP)
        any_hnn.eval()

        q_pred, p_pred = self.linear_integrator.integrate( any_hnn, self._phase_space, MD_iterations, nsamples_cur, self._tau_cur)

        self._tau_cur = MD_parameters.tau_short

        self._phase_space.set_q(torch.unsqueeze(q_list[0], dim=0))
        self._phase_space.set_p(torch.unsqueeze(p_list[0], dim=0))

        test_data = self._phase_space.read( filename, nsamples=MD_parameters.select_nsamples)
        q_truth, p_truth = self._data_io_obj.phase_space2label( test_data, self.linear_integrator, self.noML_hamiltonian)

        print('predict',q_pred, p_pred)
        print('truth', q_truth, p_truth)

        return q_pred, p_pred



    def step(self,phase_space,pb,tau):
        pairwise_hnn.eval()
        q_list_predict, p_list_predict = self.linear_integrator.integrate(**self._state)
        return q_list_predict,p_list_predict
