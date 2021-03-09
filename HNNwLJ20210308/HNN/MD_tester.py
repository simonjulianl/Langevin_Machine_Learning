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


class MD_tester:

    _obj_count = 0

    def __init__(self, linear_integrator_obj, any_HNN_obj, phase_space, path, load_path):

        MD_tester._obj_count += 1
        assert (MD_tester._obj_count == 1), type(self).__name__ + " has more than one object"

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
        _test_data = self._data_io_obj.loadq_p('test')
        self.test_data = self._data_io_obj.hamiltonian_testset(_test_data)
        self.test_data = self.test_data[:3]
        print('n. of data', self.test_data.shape)
        end_data_load = time.time()

        self._device = ML_parameters.device

        if ML_parameters.optimizer == 'Adam':
            self._opt = optim.Adam(self.any_network.parameters(), lr = ML_parameters.lr)

        elif ML_parameters.optimizer == 'SGD':
            self._opt = optim.SGD(self.any_network.parameters(), lr=ML_parameters.lr)

        checkpoint = torch.load(load_path)[0]
        # print(checkpoint)

        # load models weights state_dict
        self.any_network.load_state_dict(checkpoint['model_state_dict'])
        print('Previously trained models weights state_dict loaded...')
        self._opt.load_state_dict(checkpoint['optimizer'])
        print('Previously trained optimizer state_dict loaded...')
        # print("Optimizer's state_dict:")
        # for var_name in self._opt.state_dict():
        #     print(var_name, "\t", self._opt.state_dict()[var_name])

    def pred_qnp(self, phase_space):

        nsamples_cur = MD_parameters.nsamples_ML
        self._tau_cur = MD_parameters.tau_long  # tau = 0.1
        MD_iterations = int( MD_parameters.tau_long / self._tau_cur)

        q_pred, p_pred = self.linear_integrator.step(self.any_HNN, phase_space, MD_iterations, nsamples_cur, self._tau_cur)

        return q_pred, p_pred

    def each_sample_step(self, q_test, p_test):

        q_list_pred = None
        p_list_pred = None

        ML_iterations = int(MD_parameters.max_ts / MD_parameters.tau_long)
        
        self.any_HNN.eval()

        for i in range(ML_iterations):

            if i == 0:

                self._phase_space.set_q(torch.unsqueeze(q_test, dim=0).to(self._device))
                self._phase_space.set_p(torch.unsqueeze(p_test, dim=0).to(self._device))

                q_pred, p_pred = self.pred_qnp(self._phase_space)

                q_list_pred = q_pred
                p_list_pred = p_pred

            else:

                self._phase_space.set_q(q_list_pred[i-1].to(self._device))
                self._phase_space.set_p(p_list_pred[i-1].to(self._device))

                q_pred, p_pred = self.pred_qnp(self._phase_space)

                q_list_pred = torch.cat((q_list_pred, q_pred))
                p_list_pred = torch.cat((p_list_pred, p_pred))


        return q_list_pred,p_list_pred

    def step(self):

        _q_test = self.test_data[:, 0]; _p_test = self.test_data[:, 1]

        nsamples, nparticle, DIM = _q_test.shape

        q_list = None
        p_list = None

        for i in range(nsamples):

            q_sample_list, p_sample_list = self.each_sample_step(_q_test[i], _p_test[i])

            if i == 0:
                q_list = q_sample_list
                p_list = p_sample_list
            else:
                q_list = torch.cat((q_list, q_sample_list), dim=1)
                p_list = torch.cat((p_list, p_sample_list), dim=1)


        return (q_list, p_list)