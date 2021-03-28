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

    ''' field_HNN class to learn dHdq and then combine with nodHdq '''

    _obj_count = 0

    def __init__(self, fields_unet, linear_integrator_obj):

        field_HNN._obj_count += 1
        assert (field_HNN._obj_count == 1),type(self).__name__ + " has more than one object"

        super().__init__()
        self.network = fields_unet
        self.linear_integrator = linear_integrator_obj

        # # append term to calculate dHdq
        super().append(lennard_jones())
        super().append(kinetic_energy(MD_parameters.mass))

        self.phi_fields_obj = phi_fields(MD_parameters.npixels, super())

    def train(self):
        self.network.train()  # pytorch network for training

    def eval(self):     # pytorch network for eval
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

        corrected_dHdq = noML_dHdq.to(ML_parameters.device) + predict.to(ML_parameters.device)  # in linear_vv code, it calculates grad potential.

        return corrected_dHdq


    def diff_tau(self):

        ''' function to prepare different time steps '''

        tau = torch.tensor([MD_parameters.tau_long])
        tau = torch.unsqueeze(tau, dim=0)
        return tau


    def euclidean_distance(self, vector1, vector2):

        ''' function to measure distance between nearest grid and particle '''

        return torch.sqrt(torch.sum(torch.pow(vector1 - vector2, 2), dim=1))  # sum DIM that is dim=1 <- nparticle x DIM

    def k_nearest_grids(self, q_list, grid_interval):

        ''' function to find grids near particle '''

        k_nearest_up_left = torch.floor(q_list / grid_interval) * grid_interval

        k_nearest_up_left_x = k_nearest_up_left[:, 0]
        k_nearest_up_left_x = k_nearest_up_left_x.reshape(-1, 1)
        k_nearest_up_left_y = k_nearest_up_left[:, 1]
        k_nearest_up_left_y = k_nearest_up_left_y.reshape(-1, 1)

        k_nearest_up_right = torch.cat([k_nearest_up_left_x, k_nearest_up_left_y + grid_interval], dim=1)
        k_nearest_down_left = torch.cat([k_nearest_up_left_x + grid_interval, k_nearest_up_left_y], dim=1)
        k_nearest_down_right = torch.cat([k_nearest_up_left_x + grid_interval, k_nearest_up_left_y + grid_interval], dim=1)

        k_nearest = torch.stack([k_nearest_up_left, k_nearest_up_right, k_nearest_down_left, k_nearest_down_right], dim=1)  # samples x 4_nearest x DIM

        return k_nearest

    def k_nearest_coord(self, k_nearest, grid_interval):

        ''' function to find index of force for grids near particle '''

        k_nearest_coord = ( MC_parameters.boxsize / 2. + k_nearest) / grid_interval
        kind = torch.round(k_nearest_coord).long()  # 4_nearest x nsamples x DIM
        return kind

    def force4nparticle(self, phase_space):

        ''' function to find 4 nearest grids and calc farce about each particle '''

        predict_fields = self.fields2cnn(phase_space, self.diff_tau())

        _q_list_in = phase_space.get_q()

        grid_interval = MC_parameters.boxsize / MD_parameters.npixels

        nsamples, nparticle, DIM = _q_list_in.shape
        k_nearest_nparticle_app = []
        k_nearest_coord_nparticle_app = []

        # Distance btw each particle and 4 nearest grids
        # 4 nearest girds coordinates
        for i in range(nparticle):

            k_nearest = self.k_nearest_grids(_q_list_in[:, i], grid_interval)

            k_nearest_up_left_distance = self.euclidean_distance(_q_list_in[:, i], k_nearest[:, 0])
            k_nearest_up_right_distance = self.euclidean_distance(_q_list_in[:, i], k_nearest[:, 1])
            k_nearest_down_left_distance = self.euclidean_distance(_q_list_in[:, i], k_nearest[:, 2])
            k_nearest_down_right_distance = self.euclidean_distance(_q_list_in[:, i], k_nearest[:, 3])

            k_nearest_distance_cat = torch.stack((k_nearest_up_left_distance, k_nearest_up_right_distance,
                                                  k_nearest_down_left_distance, k_nearest_down_right_distance), dim=-1)
            k_nearest_nparticle_app.append(k_nearest_distance_cat)

            kind = self.k_nearest_coord(k_nearest, grid_interval)
            k_nearest_coord_nparticle_app.append(kind)

        k_nearest_distance_nparticle = torch.stack(k_nearest_nparticle_app, dim=1)  # nsample x nparticle x k_nearest
        k_nearest_coord_nparticle = torch.stack(k_nearest_coord_nparticle_app, dim=1) #nsample x nparticle x k_nearest x DIM

        predict_app = []

        for z in range(nsamples): # take 4 nearest forces

            # shape = DIM x nparticles
            predict_up_left = predict_fields[z, :, k_nearest_coord_nparticle[z, :, 0, 0], k_nearest_coord_nparticle[z, :, 0, 1]]
            predict_up_right = predict_fields[z, :, k_nearest_coord_nparticle[z, :, 1, 0], k_nearest_coord_nparticle[z, :, 1, 1]]
            predict_down_left = predict_fields[z, :, k_nearest_coord_nparticle[z, :, 2, 0], k_nearest_coord_nparticle[z, :, 2, 1]]
            predict_down_right = predict_fields[z, :, k_nearest_coord_nparticle[z, :, 3, 0], k_nearest_coord_nparticle[z, :, 3, 1]]

            predict_cat = torch.stack((predict_up_left, predict_up_right, predict_down_left, predict_down_right), dim=-1) # shape = DIM x nparticles x k_nearest
            predict_app.append(predict_cat)

        predict_k_nearest_force = torch.stack(predict_app)   # sample x  DIM   x npartilce x k_nearest

        z_l = torch.sum(1. / k_nearest_distance_nparticle, dim=-1)
        z_l = z_l.unsqueeze(dim=1)

        k_nearest_distance_nparticle_unsqueeze = k_nearest_distance_nparticle.unsqueeze(dim=1)  # nsample x 1 x nparticle x k_nearest

        predict_each_particle = 1. / z_l * (torch.sum(1. / k_nearest_distance_nparticle_unsqueeze * predict_k_nearest_force, dim=-1))  # nsample x DIM x nparticle
        predict_each_particle = predict_each_particle.permute((0, 2, 1))  # nsample x nparticle x DIM

        if (k_nearest_distance_nparticle_unsqueeze < 0.001).any():
            index = torch.where(k_nearest_distance_nparticle < 0.001)  # nsample x nparticle x k_nearest
            predict_k_vv_nearest_force = predict_k_nearest_force[index[0], :, index[1], index[2]]  # sample x  DIM   x npartilce x k_nearest
            predict_each_particle[index[0], index[1]] = predict_k_vv_nearest_force

        return predict_each_particle


    def fields2cnn(self,phase_space, tau):

        ''' function to feed phi field and p_field as input to cnn

        parameters
        tau : torch.tensor
                different time steps

        Returns
        ----------
        predict fields each grids on x, y
        '''

        phi_field = self.phi_field4cnn(phase_space)
        p_field = self.p_field4cnn()
        self.predict_fields = self.network(phi_field, p_field, tau)
        print('predict shape', self.predict_fields.shape)
        return self.predict_fields


    def phi_field4cnn(self, phase_space):

        ''' function to prepare phi field as input in cnn '''

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

        ''' function to apply pbc pad to p field '''

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

        ''' function to prepare p field as input in cnn '''

        self.phi_field_pbc_in = self.p_field_pbc_padding(self._phi_field_in)
        # self.phi_fields_obj.show_gridimg(phi_field_pbc_in[:][0], '(t)')

        self.phi_field_pbc_nx = self.p_field_pbc_padding(self._phi_field_nx)
        # self.phi_fields_obj.show_gridimg(phi_field_pbc_nx[:][0], '(t+$\delta$t)')

        self.momentum_fields_obj = momentum_fields(self.phi_field_pbc_in, self.phi_field_pbc_nx)

        flow_vectors = self.momentum_fields_obj.p_field()
        print('before crop', flow_vectors.shape)
        # momentum_fields_obj.visualize_flow_file(flow_vectors)
        flow_vectors_crop = flow_vectors[:,:,1:-1,1:-1]
        print('after crop', flow_vectors_crop.shape)

        return flow_vectors_crop