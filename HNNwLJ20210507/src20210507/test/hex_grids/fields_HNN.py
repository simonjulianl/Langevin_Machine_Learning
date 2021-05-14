import torch
import itertools
from hamiltonian            import hamiltonian
from lennard_jones          import lennard_jones
from kinetic_energy         import kinetic_energy
from phi_fields                  import phi_fields
import time

class fields_HNN(hamiltonian):
    ''' pairwise_HNN class to learn dHdq and then combine with nodHdq '''

    _obj_count = 0

    def __init__(self, fields_MLP1, fields_MLP2, integrator):

        super(fields_HNN,self).__init__() # python 2 ; python 3 super().__init__()

        fields_HNN._obj_count += 1
        assert (fields_HNN._obj_count == 1),type(self).__name__ + " has more than one object"

        self.netA = fields_MLP1   # network for dHdq1
        self.netB = fields_MLP2   # network for dHdq2

        # # append term to calculate dHdq
        super().append(lennard_jones())
        super().append(kinetic_energy())

        self.integrator         = integrator
        self.phi_fields         = phi_fields(super())

        self.tau_short       = None # check give tau or not
        self.tau_long      = None # check give tau or not HK add

        print('fields_HNN initialized ')
        self.dt = 0
    # ===================================================
    def set_tau_short(self, tau_short):  # HK
        self.tau_short = tau_short
    # ===================================================
    def set_tau_long(self, tau_long): # HK
        self.tau_long = tau_long
    # ===================================================
    def get_netlist(self): # nochange
        return [self.netA,self.netB]
    # ===================================================
    def net_parameters(self): # nochange
        ''' To give multiple parameters to the optimizer,
            concatenate lists of parameters from two models '''
        return itertools.chain(self.netA.parameters(), self.netB.parameters())
    # ===================================================
    def train(self):
        '''pytorch network for training'''
        self.netA.train()
        self.netB.train()
    # ===================================================
    def eval(self):
        ''' pytorch network for eval '''
        self.netA.eval()
        self.netB.eval()
    # ===================================================
    def requires_grad_false(self):

        for param in self.netA.parameters():
            param.requires_grad = False

        for param in self.netB.parameters():
            param.requires_grad = False
    # ===================================================
    def dHdq1(self, phase_space): # nochange
        # use this to update p in first step of velocity verlet
        return self.dHdq_all(phase_space, self.netA)
    # ===================================================
    def dHdq2(self, phase_space): # nochange
        # use this to update p in third step of velocity verlet
        return self.dHdq_all(phase_space, self.netB)
    # ===================================================
    def dHdq_all(self, phase_space, net): # nochange
        ''' function to calculate dHdq = noML_dHdq + residual ML_dHdq

        Parameters
        ----------
        phase_space : contains q_list, p_list as input
                q_list shape is [nsamples, nparticle, DIM]
        net         : pass fields_hnn

        Returns
        ----------
        corrected_dHdq : torch.tensor
                shape is [nsamples,nparticle,DIM]

        '''
        nsamples, nparticle, DIM = phase_space.get_q().shape

        noML_dHdq = super().dHdq(phase_space)
        # noML_dHdq shape is [nsamples, nparticle, DIM]

        x = self.make_fields(phase_space, self.integrator, self.tau_short)
        # x.shape = [ nsamples, nparticle, grids18 + grids18 ]

        x = torch.reshape(x, (nsamples * nparticle, -1)).double() # HK
        # x.shape = [ nsamples*nparticle, grids18 + grids18]

        start = time.time()
        out = net(x)
        # predict.shape = [nsamples*nparticle, 2]
        end = time.time()
        self.dt += (end-start)

        predict = torch.reshape(out, (nsamples, nparticle, DIM))

        corrected_dHdq = noML_dHdq + predict
        # corrected_dHdq.shape = [nsamples, nparticle, DIM]

        # corrected_dHdq = noML_dHdq # for testing if can run

        return corrected_dHdq

    # ===================================================
    def make_fields(self, phase_space, integrator, tau_short):

        q_list_original = phase_space.get_q()
        p_list_original = phase_space.get_p()

        # print(q_list_original)
        # grids_list  = self.phi_fields.build_grid_list(phase_space)
        # # shape is [gridL * gridL , DIM=(x,y)]
        #
        # self.phi_fields.show_grids_nparticles(grids_list, q_list_original)

        grids18_1 = self.phi_fields.gen_phi_fields(phase_space)
        # shape is [ nsamples, nparticle, grids18 ]

        qp_list, crash_idx = integrator.one_step( super(), phase_space, tau_short)
        # qp_list shape, [nsamples, (q,p)=2, nparticle, DIM]
        # integrate one step update phase space

        grids18_2 = self.phi_fields.gen_phi_fields(phase_space)
        # shape is [ nsamples, nparticle, grids18 ]

        # copy back to original
        phase_space.set_q(q_list_original)
        phase_space.set_p(p_list_original)

        grids36 = torch.cat((grids18_1,grids18_2),dim=-1)
        # grids36.shape = shape is [ nsamples, nparticle, grids18 + grids18 ]

        return grids36