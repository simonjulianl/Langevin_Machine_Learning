from MC_parameters import MC_parameters
from MD_parameters import MD_parameters
from ML_parameters import ML_parameters
from hamiltonian import hamiltonian
from hamiltonian.lennard_jones import lennard_jones
from hamiltonian.kinetic_energy import kinetic_energy
from fields.phi_fields import phi_fields
from fields.momentum_fields import momentum_fields
import torch

class field_HNN(hamiltonian):

    _obj_count = 0

    def __init__(self, network, linear_integrator_obj):

        field_HNN._obj_count += 1
        assert (field_HNN._obj_count == 1),type(self).__name__ + " has more than one object"

        super().__init__()
        self.network = network
        self.linear_integrator = linear_integrator_obj

        # # append term to calculate dHdq
        super().append(lennard_jones())
        super().append(kinetic_energy(MD_parameters.mass))


        self.phi_fields_obj = phi_fields(MD_parameters.npixels, super())

    def train(self):
        self.network.train()  # pytorch network for training

    def eval(self):
        self.network.eval()

    def dHdq(self, phase_space):

        q_list = phase_space.get_q()
        p_list = phase_space.get_p()

        noML_dHdq = super().dHdq(phase_space)

        phase_space.set_q(q_list)
        phase_space.set_p(p_list)

        # predict_fields to calc predict each particle
        # predict =  #xxxx
        predict = self.force4nparticle(phase_space)

        corrected_dHdq = noML_dHdq.to(ML_parameters.device) + predict  # in linear_vv code, it calculates grad potential.

        return corrected_dHdq

    def fields2cnn(self,phase_space, tau):

        phi_field = self.phi_field4cnn(phase_space)
        p_field = self.p_field4cnn()
        self.predict_fields = self.network(phi_field, p_field, tau)
        print('predict shape', self.predict_fields.shape)
        return self.predict_fields

    def torch_tau(self):
        tau = torch.tensor([MD_parameters.tau_long])
        tau = torch.unsqueeze(tau, dim=0)
        return tau

    def euclidean_distance(self, vector1, vector2):
        return torch.sqrt(torch.sum(torch.pow(vector1 - vector2, 2)))

    # find 4 nearest neighbor grids and calc farce about each particle
    def force4nparticle(self, phase_space):

        predict_fields = self.fields2cnn(phase_space, self.torch_tau())

        _q_list_in = phase_space.get_q()

        print(_q_list_in)
        grid_interval = MC_parameters.boxsize / MD_parameters.npixels
        print('interval', grid_interval)
        nsamples, nparticle, DIM = _q_list_in.shape

        for i in range(nparticle):

            print(i)
            k_nearest_up_left = torch.floor(_q_list_in[:,i] / grid_interval ) * grid_interval

            k_nearest_up_left_x = k_nearest_up_left[:,0]
            k_nearest_up_left_y = k_nearest_up_left[:,1]

            k_nearest_up_right = torch.tensor([[k_nearest_up_left_x, k_nearest_up_left_y + grid_interval]])
            k_nearest_down_left = torch.tensor([[k_nearest_up_left_x + grid_interval, k_nearest_up_left_y]])
            k_nearest_down_right = torch.tensor([[k_nearest_up_left_x + grid_interval, k_nearest_up_left_y + grid_interval]])

            k_nearest_up_left_distance = self.euclidean_distance(_q_list_in[:,i], k_nearest_up_left)
            k_nearest_up_right_distance = self.euclidean_distance(_q_list_in[:,i], k_nearest_up_right)
            k_nearest_down_left_distance = self.euclidean_distance(_q_list_in[:,i], k_nearest_down_left)
            k_nearest_down_right_distance = self.euclidean_distance(_q_list_in[:,i], k_nearest_down_right)

            z_distance = 1. / k_nearest_up_left_distance + 1. / k_nearest_up_right_distance + 1. / k_nearest_down_left_distance + 1. / k_nearest_down_right_distance

            up_left_coord = ( MC_parameters.boxsize / 2 + k_nearest_up_left ) / grid_interval
            # up_left_index = up_left_coord[:,0] * MD_parameters.npixels + up_left_coord[:,1]

            up_right_coord = ( MC_parameters.boxsize / 2 + k_nearest_up_right ) / grid_interval
            # up_right_index = up_right_coord[:,0] * MD_parameters.npixels + up_right_coord[:,1]

            down_left_coord = ( MC_parameters.boxsize / 2 + k_nearest_down_left ) / grid_interval
            # down_left_index = down_left_coord[:,0] * MD_parameters.npixels + down_left_coord[:,1]

            down_right_coord = ( MC_parameters.boxsize / 2 + k_nearest_down_right ) / grid_interval
            # down_right_index = down_right_coord[:,0] * MD_parameters.npixels + down_right_coord[:,1]

            print(up_left_coord, up_right_coord, down_left_coord, down_right_coord)
            # print(up_left_index, up_right_index, down_left_index, down_right_index)

            print(predict_fields.shape)
            predict_up_left = predict_fields[:,:,int(up_left_coord[:,0]),int(up_left_coord[:,1])]
            predict_up_right = predict_fields[:,:,int(up_right_coord[:,0]),int(up_right_coord[:,1])]
            predict_down_left = predict_fields[:,:,int(down_left_coord[:,0]),int(down_left_coord[:,1])]
            predict_down_right = predict_fields[:,:,int(down_right_coord[:,0]),int(down_right_coord[:,1])]


            predict_each_particle = ( 1. / z_distance ) * (1. / k_nearest_up_left_distance * predict_up_left
                                                           + 1. / k_nearest_up_right_distance * predict_up_right
                                                           + 1. / k_nearest_down_left_distance * predict_down_left
                                                           + 1. / k_nearest_down_right_distance * predict_down_right )

            print('predict', predict_each_particle)


        return predict_fields4particle

    def phi_field4cnn(self, phase_space):

        _q_list_in = phase_space.get_q().cpu()
        _p_list_in = phase_space.get_p().cpu()

        print('inital q, p', _q_list_in, _p_list_in)

        phase_space.set_q(_q_list_in)
        # self.phi_fields_obj.show_grid_nparticles(_q_list_in,'position each particle')

        self._phi_field_in = self.phi_fields_obj.phi_field(phase_space)
        print('show in', self._phi_field_in.shape)
        # self.phi_fields_obj.show_gridimg(self._phi_field_in, '(t)')

        nsamples_cur = MD_parameters.nsamples_batch
        tau_cur = MD_parameters.tau_short
        MD_iterations = int(MD_parameters.tau_short / MD_parameters.tau_short)
        print('next state for phi field : nsamples_cur, tau_cur, MD_iterations')
        print(nsamples_cur, tau_cur, MD_iterations)

        phase_space.set_q(_q_list_in)
        phase_space.set_p(_p_list_in)

        _q_list_nx, _p_list_nx = self.linear_integrator.step(super(), phase_space, MD_iterations, nsamples_cur, tau_cur)
        _q_list_nx = _q_list_nx[-1].type(torch.float64)  # only take the last from the list
        _p_list_nx = _p_list_nx[-1].type(torch.float64)

        # print('next q, p', _q_list_nx, _p_list_nx)

        phase_space.set_q(_q_list_nx)
        # phase_space.set_p(_p_list_nx)
        self._phi_field_nx = self.phi_fields_obj.phi_field(phase_space) # nsamples x npixels x npixels
        print('show nx', self._phi_field_nx.shape)
        # self.phi_fields_obj.show_gridimg(self._phi_field_nx, '(t+$\delta$t)')
        print(self._phi_field_nx.shape)

        self._phi_field_in = torch.unsqueeze(self._phi_field_in, dim=1) # nsamples x channel x npixels x npixels
        self._phi_field_nx = torch.unsqueeze(self._phi_field_nx, dim=1)
        phi_field_cat = torch.cat((self._phi_field_in, self._phi_field_nx), dim=1)
        print('network shape', phi_field_cat.shape)

        return phi_field_cat  # nsamples x 2 x npixels x npixels

    def p_field_pbc_padding(self, fields):
        print('fields shape', fields.shape)
        v = torch.zeros(MD_parameters.npixels, )
        v[0] = 1

        v_flip = torch.flip(v, dims=(0,))

        v = torch.unsqueeze(v, dim=0)
        v_flip = torch.unsqueeze(v_flip, dim=0)

        b = torch.cat([v_flip, torch.eye(MD_parameters.npixels), v])
        b_t = torch.transpose(b, 0, 1)

        b = b.unsqueeze(0).repeat(MD_parameters.nsamples_batch, 1, 1)
        b_t = b_t.unsqueeze(0).repeat(MD_parameters.nsamples_batch, 1, 1)

        b = b.unsqueeze(1)
        b_t = b_t.unsqueeze(1)

        x = torch.matmul(torch.matmul(b, fields), b_t)
        print('pbc', x.shape)
        return x

    def p_field4cnn(self):

        phi_field_pbc_in = self.p_field_pbc_padding(self._phi_field_in)
        # self.phi_fields_obj.show_gridimg(phi_field_pbc_in[:][0], '(t)')

        phi_field_pbc_nx = self.p_field_pbc_padding(self._phi_field_nx)
        # self.phi_fields_obj.show_gridimg(phi_field_pbc_nx[:][0], '(t+$\delta$t)')

        momentum_fields_obj = momentum_fields(phi_field_pbc_in, phi_field_pbc_nx)
        flow_vectors = momentum_fields_obj.p_field()
        print('before crop', flow_vectors.shape)
        # momentum_fields_obj.visualize_flow_file(flow_vectors)
        flow_vectors_crop = flow_vectors[:,:,1:-1,1:-1]
        print('after crop', flow_vectors_crop.shape)

        return flow_vectors_crop