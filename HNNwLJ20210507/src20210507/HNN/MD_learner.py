from HNN.loss                 import qp_MSE_loss
from HNN.checkpoint           import checkpoint
import time
import math

import torch

class MD_learner:

    ''' MD_learner class to help train, validate, retrain, and save '''

    _verbose_interval = 10     # adjust this to show results
    _checkpoint_interval = 50  # adjust this to checkpoint
    _obj_count = 0

    def __init__(self, linear_integrator_obj, 
                 any_HNN_obj, phase_space, opt, 
                 sch, data_loader, load_model_file=None): 

        '''
        Parameters
        ----------
        linear_integrator_obj : use for integrator using large time step
        any_HNN_obj : pass any HNN object to this container
        phase_space : contains q_list, p_list as input
                    q list shape is [nsamples, nparticle, DIM]
        opt         : create one optimizer from two models parameters
        sch         : lr decay 0.99 every 100 epochs
        data_loader : DataLoaders on Custom Datasets
                 two tensors contain train and valid data
                 each shape is [nsamples, 2, niter, nparticle, DIM] , here 2 is (q,p)
                 niter is initial and append strike iter so that 2
        load_model_file : file for save or load them
                 default is None
        '''

        MD_learner._obj_count += 1
        assert (MD_learner._obj_count == 1), type(self).__name__ + " has more than one object"

        self.linear_integrator = linear_integrator_obj
        self.any_HNN = any_HNN_obj
        self.data_loader = data_loader

        self._phase_space = phase_space
        self._opt = opt
        self._sch = sch
        self.chk_pt = checkpoint(self.any_HNN.get_netlist(), self._opt, self._sch)

        if load_model_file is not None: self.chk_pt.load_checkpoint(load_model_file)

        self._loss      = qp_MSE_loss
        # self.batch_size = self.data_loader.batch_size

        self.tau_cur = self.data_loader.data_set.train_set.data_tau_long
        boxsize      = self.data_loader.data_set.train_set.data_boxsize

        self._phase_space.set_boxsize(boxsize)

        print('MD_learner initialized : tau_cur ',self.tau_cur,' boxsize ',boxsize)

    # ===================================================
    def train_one_epoch(self):
        ''' function to train one epoch'''

        self.any_HNN.train()
        train_loss  = 0.
        train_qloss = 0.
        train_ploss = 0.

        for step, (input, label) in enumerate(self.data_loader.train_loader):

            self._opt.zero_grad()
            # clear out the gradients of all variables in this optimizer (i.e. w,b)

            # input shape, [nsamples, (q,p)=2, nparticle, DIM]
            self._phase_space.set_q(input[:, 0, :, :])
            self._phase_space.set_p(input[:, 1, :, :])

            qp_list, crash_idx = self.linear_integrator.one_step(self.any_HNN, self._phase_space, self.tau_cur)
            # qp_list shape, [nsamples, (q,p)=2, nparticle, DIM]
            q_predict = qp_list[:, 0, :, :]; p_predict = qp_list[:, 1, :, :]
            # q_predict shape, [nsamples, nparticle, DIM]
            print('predict',q_predict)
            q_label   = label[:, 0, :, :]  ; p_label   = label[:, 1, :, :]
            # q_label shape, [nsamples, nparticle, DIM]
            print('label',q_label)
            train_predict = (q_predict, p_predict)
            train_label = (q_label, p_label)

            loss,qloss,ploss = self._loss(train_predict, train_label) 

            loss.backward()
            # backward pass : compute gradient of the loss wrt models parameters

            self._opt.step()

            train_loss  += loss.item()   # get the scalar output 
            train_qloss += qloss.item()  # get the scalar output
            train_ploss += ploss.item()  # get the scalar output

        quit()
        return train_loss / (step+1), train_qloss / (step+1), train_ploss / (step+1)

    # ===================================================
    def valid_one_epoch(self): 
        ''' function to valid one epoch'''

        self.any_HNN.eval()
        valid_loss  = 0.
        valid_qloss = 0.
        valid_ploss = 0.

        for step, (input, label) in enumerate(self.data_loader.val_loader):

            # input shape, [nsamples, (q,p)=2, nparticle, DIM]
            self._phase_space.set_q(input[:, 0, :, :])
            self._phase_space.set_p(input[:, 1, :, :])

            qp_list, crash_idx = self.linear_integrator.one_step(self.any_HNN, self._phase_space, self.tau_cur)
            # qp_list shape, [nsamples, (q,p)=2, nparticle, DIM]
            q_predict = qp_list[:, 0, :, :]; p_predict = qp_list[:, 1, :, :] # q_predict shape, [nsamples, nparticle, DIM]

            q_label   = label[:, 0, :, :]  ; p_label   = label[:, 1, :, :]
            # q_label shape, [nsamples, nparticle, DIM]

            valid_predict = (q_predict, p_predict)
            valid_label = (q_label, p_label)

            loss,qloss,ploss = self._loss(valid_predict, valid_label)

            valid_loss  += loss.item()   # get the scalar output
            valid_qloss += qloss.item()  # get the scalar output
            valid_ploss += ploss.item()  # get the scalar output

        return valid_loss / (step+1), valid_qloss / (step+1), valid_ploss / (step+1)

    # ===================================================
    def nepoch(self, nepoch, write_chk_pt_basename, write_loss_filename):

        ''' function to train and valid more than one epoch

        parameters
        -------
        write_chk_pt_basename : filename to save checkpoints 
        write_loss_filename   : path + filename to save loss values

        Returns
        -------
        float
            train loss, valid loss every epoch
        '''

        text = ''

        for e in range(0, nepoch ):

            start_epoch = time.time()

            train_loss,train_qloss,train_ploss = self.train_one_epoch()

            self._sch.step() # learning

            with torch.no_grad(): # reduce memory consumption for computations  
                valid_loss, valid_qloss, valid_ploss = self.valid_one_epoch()

            end_epoch = time.time()

            if e%MD_learner._checkpoint_interval == 0:
                this_filename = write_chk_pt_basename + str(e+1) + '.pth'
                self.chk_pt.save_checkpoint(this_filename)

            train_dq = math.sqrt(train_qloss)
            train_dp = math.sqrt(train_ploss)

            valid_dq = math.sqrt(valid_qloss)  
            valid_dp = math.sqrt(valid_ploss) 

            dt = end_epoch - start_epoch

            boxsz = self._phase_space.get_boxsize()
            text = text + str(e+1) + ' ' + str(train_loss)  + ' ' + str(valid_loss)  + \
                   ' ' + str(train_dq/boxsz)  + ' ' + str(valid_dq/boxsz) + \
                   ' ' + str(train_dq) + ' '  + str(valid_dq) + \
                   ' ' + str(train_dp) + ' '  + str(valid_dp) + ' ' + str(self._opt.param_groups[0]['lr']) +\
                   ' ' + str(dt) + '\n'

            if e%MD_learner._verbose_interval==0:
                print('{} epoch:'.format(e+1), 'train_loss:{:.6f}'.format(train_loss), 
                      'valid_loss:{:.6f}'.format(valid_loss), ' each epoch time:{:.5f}'.format(dt))
                print('optimizer lr {:.5f}'.format(self._opt.param_groups[0]['lr']),
                      ' boxsize {:.5f}'.format(boxsz),' train_dq/boxsize {:.6f}'.format(train_dq/boxsz),
                      ' valid_dq/boxsize {:.6f}'.format(valid_dq/boxsz), ' train_dp {:.6f}'.format(train_dp),
                      ' valid_dp {:.6f}'.format(valid_dp), flush=True )

                self.save_loss_curve(text, write_loss_filename) 


    # ===================================================
    def save_loss_curve(self, text, write_loss_filename):
        ''' function to save the loss every epoch '''

        with open(write_loss_filename, 'w') as fp:
            fp.write(text)
        fp.close()

