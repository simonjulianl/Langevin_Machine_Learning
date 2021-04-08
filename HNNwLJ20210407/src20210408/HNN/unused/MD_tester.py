# from .data_io import data_io
from ._checkpoint import _checkpoint
from MC_parameters import MC_parameters
from MD_parameters import MD_parameters
from ML_parameters import ML_parameters
import torch
# import psutil
import gzip
import pickle
import os

class MD_tester:

    ''' MD_tester class to help predict trajectory at initial state '''

    _obj_count = 0

    def __init__(self, linear_integrator_obj, any_HNN_obj, phase_space, init_test_path, load_path, crash_filename=None):

        '''
        Parameters
        ----------
        init_test_path : string
                folder name
        load_path : string
                load saved model
        crash_filename : str, optional
                default is None

        Returns
        ----------
        predicted q_list : torch.tensor
        predicted p_list : torch.tensor
        q_list, p_list to torch.zeros if crash_index exist
        '''

        MD_tester._obj_count += 1
        assert (MD_tester._obj_count == 1), type(self).__name__ + " has more than one object"

        self.linear_integrator = linear_integrator_obj
        self.any_HNN = any_HNN_obj
        self.any_network = any_HNN_obj.network
        self.noML_hamiltonian = super(type(any_HNN_obj), any_HNN_obj)

        print('-- hi terms -- ', self.noML_hamiltonian.hi())

        self._phase_space = phase_space
        self._data_io_obj = data_io(init_test_path)

        print("============ start data loaded ===============")
        self.q_test, self.p_test = self._data_io_obj.qp_dataset('test', shuffle = False)
        print('n. of valid data reshape ', self.q_test.shape, self.p_test.shape)

        self._device = ML_parameters.device

        self._opt = ML_parameters.opt.create(self.any_network.parameters())
        print(ML_parameters.opt.name())


        # load models weights state_dict
        self._checkpoint = _checkpoint(self.any_network, self._opt, current_epoch = 0)
        self._checkpoint.load_checkpoint(load_path)


    def pred_qnp(self, phase_space):

        ''' function to predict state q, p at one step using ML

        Parameters
        ----------
        nsamples_cur : int
                1 : load input each sample in pair wise HNN
                n : load input batch samples in optical flow HNN
        tau_cur : large time step for ML
        MD_iterations :
                1 : predict single step

        Returns
        ----------
        predicted q_list : torch.tensor
                predict q list at single step
        predicted p_list : torch.tensor
                predict p list at single step
        '''

        nsamples_cur = MD_parameters.nsamples_ML
        self._tau_cur = MD_parameters.tau_long  # tau = 0.1
        MD_iterations = int(MD_parameters.tau_long / self._tau_cur)

        q_pred, p_pred = self.linear_integrator.step(self.any_HNN, phase_space, MD_iterations, nsamples_cur, self._tau_cur)

        return q_pred, p_pred

    def each_sample_step(self, q_test, p_test):

        ''' function to predict trajectory from initial state each sample

        Parameters
        ----------
        ML_iterations : int
                n iterations with tau_long up to maximum time
        crash_index : index of q list that is crashed
        bool_ : debug pbc
                if True, add crash_index to the list

        Returns
        ----------
        predicted q_list : torch.tensor
        predicted p_list : torch.tensor
        q_list, p_list to torch.zeros if crash_index exist
        '''

        q_list_pred = None
        p_list_pred = None
        crash_index = []

        ML_iterations = int(MD_parameters.max_ts / MD_parameters.tau_long)

        self.any_HNN.eval()

        for i in range(ML_iterations):

            # print('ML_iterations', i)

            if i == 0:
                # predict one step from initial state
                self._phase_space.set_q(torch.unsqueeze(q_test, dim=0).to(self._device))
                self._phase_space.set_p(torch.unsqueeze(p_test, dim=0).to(self._device))

                q_pred, p_pred = self.pred_qnp(self._phase_space)

                q_list_pred = q_pred
                p_list_pred = p_pred

            else:
                # predict one step from previous predicted state
                self._phase_space.set_q(q_list_pred[i - 1].to(self._device))
                self._phase_space.set_p(p_list_pred[i - 1].to(self._device))

                q_pred, p_pred = self.pred_qnp(self._phase_space)

                bool1_ = self._phase_space.debug_pbc_bool(q_pred, MC_parameters.boxsize)
                bool2_ = self._phase_space.debug_nan_bool(q_pred, p_pred)

                # if any detect debug then add index to the list
                if bool1_.any() == True or bool2_ is not None:
                    print('bool true', i)
                    crash_index.append(i)

                q_list_pred = torch.cat((q_list_pred, q_pred))
                p_list_pred = torch.cat((p_list_pred, p_pred))

        if crash_index:  # if exist index that is debug, then make state zeros

            self.q_crash_before_pred = q_list_pred[ML_iterations - len(crash_index) - 1]
            self.p_crash_before_pred = p_list_pred[ML_iterations - len(crash_index) - 1]

            q_list_pred = torch.zeros([])
            p_list_pred = torch.zeros([])

        return q_list_pred, p_list_pred

    def step(self, filename):

        ''' function to save trajectory nsamples if no crashed state q
            otherwise return state q, p before crashed

        Parameters
        ----------
        ML_iteration_batch : int
                the num. of saved files
        filename : string
                save files every ML_iteration_batch
                discard saved files if crashed q list exist

        Returns
        ----------
        q_crash_before_pred_app : torch.tensor
                q list before crash state
        p_crash_before_pred_app : torch.tensor
                p list before crash state
        '''

        nsamples, nparticle, DIM = self.q_test.shape

        qp_list = []
        q_crash_before_pred_app = []
        p_crash_before_pred_app = []

        ML_iteration_batch = MD_parameters.ML_iteration_batch

        for i in range(nsamples):

            print('nsamples', i)

            q_sample_list, p_sample_list = self.each_sample_step(self.q_test[i], self.p_test[i])
            print(q_sample_list.shape, p_sample_list.shape)

            if q_sample_list.shape and p_sample_list.shape:
                qp_stack = torch.stack((q_sample_list, p_sample_list))
                qp_list.append(qp_stack)

            else:  # torch.zeros([]) : q_sample_list, p_sample_list that crash_index exist

                q_crash_before_pred_app.append(self.q_crash_before_pred)
                p_crash_before_pred_app.append(self.p_crash_before_pred)


            mem_usage = psutil.virtual_memory()

            print('totol memory :', bytes2human(mem_usage[0]))
            print('available memory:', bytes2human(mem_usage[1]))
            print('memory % used:', psutil.virtual_memory()[2])

            if i % ML_iteration_batch == ML_iteration_batch - 1:

                print('save file {}'.format(i))
                with gzip.open(filename + '_{}.pt'.format(i // ML_iteration_batch), 'wb') as handle:  # overwrites any existing file
                    pickle.dump(qp_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    handle.close()

                del qp_list
                qp_list = []

        if q_crash_before_pred_app:

            for z in range(int(nsamples / ML_iteration_batch)):
                os.remove(filename + '_{}.pt'.format(z))

            q_crash_before_pred_app = torch.cat(q_crash_before_pred_app, dim=0)
            p_crash_before_pred_app = torch.cat(p_crash_before_pred_app, dim=0)

        return q_crash_before_pred_app, p_crash_before_pred_app


