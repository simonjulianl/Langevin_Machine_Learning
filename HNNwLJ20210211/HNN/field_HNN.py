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
        corrected_dHdq = noML_dHdq.to(ML_parameters.device) + predict  # in linear_vv code, it calculates grad potential.

        return corrected_dHdq

    def fields2cnn(self,phase_space):

        phi_field = self.phi_field4cnn(phase_space)
        p_field = self.p_field4cnn()
        predict_fields = self.network(phi_field, p_field)

    # find the closest grid to particles and calc farce about each particle
    def force_each_particle(self, phase_space):

        _q_list_in = phase_space.get_q()
        _p_list_in = phase_space.get_p()

        


    def phi_field4cnn(self, phase_space):

        _q_list_in = phase_space.get_q()
        _p_list_in = phase_space.get_p()

        self.phi_fields_obj = phi_fields(MD_parameters.npixels, super())

        phase_space.set_q(_q_list_in)
        # self.phi_fields_obj.show_grid_nparticles(_q_list_in,'hi')

        self._phi_field_in = self.phi_fields_obj.phi_field(phase_space)
        print('show in', self._phi_field_in.shape)
        # self.phi_fields_obj.show_gridimg(self._phi_field_in)

        nsamples_cur = MD_parameters.nsamples
        tau_cur = MD_parameters.tau_long
        MD_iterations = int(MD_parameters.tau_short / MD_parameters.tau_short)

        phase_space.set_q(_q_list_in)
        phase_space.set_p(_p_list_in)

        _q_list_nx, _p_list_nx = self.linear_integrator.step(super(), phase_space, MD_iterations, nsamples_cur, tau_cur)
        _q_list_nx = _q_list_nx[-1].type(torch.float64)  # only take the last from the list
        _p_list_nx = _p_list_nx[-1].type(torch.float64)

        phase_space.set_q(_q_list_nx)
        # phase_space.set_p(_p_list_nx)
        self._phi_field_nx = self.phi_fields_obj.phi_field(phase_space) # nsamples x npixels x npixels
        print('show nx', self._phi_field_nx.shape)
        # self.phi_fields_obj.show_gridimg(self._phi_field_nx)

        self._phi_field_in = torch.unsqueeze(self._phi_field_in, dim=1) # nsamples x channel x npixels x npixels
        self._phi_field_nx = torch.unsqueeze(self._phi_field_nx, dim=1)
        phi_field_cat = torch.cat((self._phi_field_in, self._phi_field_nx), dim=1)
        print('network shape', phi_field_cat.shape)

        return phi_field_cat  # nsamples x 2 x npixels x npixels

    def p_field_pbc_padding(self, fields):

        v = torch.zeros(MD_parameters.npixels, )
        v[0] = 1

        v_flip = torch.flip(v, dims=(0,))

        v = torch.unsqueeze(v, dim=0)
        v_flip = torch.unsqueeze(v_flip, dim=0)

        b = torch.cat([v_flip, torch.eye(MD_parameters.npixels), v])
        b_t = torch.transpose(b, 0, 1)

        b = b.unsqueeze(0).repeat(MD_parameters.nsamples, 1, 1)
        b_t = b_t.unsqueeze(0).repeat(MD_parameters.nsamples, 1, 1)

        x = torch.matmul(torch.matmul(b, fields), b_t)
        print('pbc', x.shape)
        return x

    def p_field4cnn(self):

        phi_field_pbc_in = self.p_field_pbc_padding(self._phi_field_in)
        # self.phi_fields_obj.show_gridimg(phi_field_pbc_in[:][0])

        phi_field_pbc_nx = self.p_field_pbc_padding(self._phi_field_nx)
        # self.phi_fields_obj.show_gridimg(phi_field_pbc_nx[:][0])

        momentum_fields_obj = momentum_fields(phi_field_pbc_in, phi_field_pbc_nx)
        flow_vectors = momentum_fields_obj.p_field()
        # momentum_fields_obj.visualize_flow_file(flow_vectors)
        flow_vectors_crop = flow_vectors[:,:,1:-1,1:-1]

        return flow_vectors_crop