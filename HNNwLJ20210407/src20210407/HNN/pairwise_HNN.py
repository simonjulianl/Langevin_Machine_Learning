import torch
import itertools
from hamiltonian.hamiltonian            import hamiltonian
from hamiltonian.lennard_jones          import lennard_jones
from hamiltonian.kinetic_energy         import kinetic_energy
from utils.get_paired_distance_indices  import get_paired_distance_indices


class pairwise_HNN(hamiltonian):
    ''' pairwise_HNN class to learn dHdq and then combine with nodHdq '''

    _obj_count = 0

    def __init__(self, pairwise_MLP1, pairwise_MLP2):

        pairwise_HNN._obj_count += 1
        assert (pairwise_HNN._obj_count == 1),type(self).__name__ + " has more than one object"

        super().__init__()
        self.netA = pairwise_MLP1   # network for dHdq1
        self.netB = pairwise_MLP2   # network for dHdq2

        # # append term to calculate dHdq
        super().append(lennard_jones())
        super().append(kinetic_energy())

        self.tau_cur = None # check give tau or not

    # ===================================================
    def set_tau(self, tau_cur):
        self.tau_cur = tau_cur

    # ===================================================
    def net_parameters(self):
        ''' To give multiple parameters to the optimizer,
            concatenate lists of parameters from two models '''

        return itertools.chain(self.netA.parameters(), self.netB.parameters())
        #return list(self.netA.parameters())+list(self.netB.parameters())

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
    def dHdq1(self, phase_space):
        # use this to update p in first step of velocity verlet
        return self.dHdq_all(phase_space,self.netA)
    # ===================================================
    def dHdq2(self, phase_space):
        # use this to update p in third step of velocity verlet
        return self.dHdq_all(phase_space,self.netB)
    # ===================================================
    def dHdq_all(self, phase_space, net):

        ''' function to calculate dHdq = noML_dHdq + residual ML_dHdq

        Parameters
        ----------
        phase_space : contains q_list, p_list as input
                q_list shape is [nsamples, nparticle, DIM]
        net         : pass netA or netB

        Returns
        ----------
        corrected_dHdq : torch.tensor
                shape is [nsamples,nparticle,DIM]

        '''
        q_list = phase_space.get_q()

        noML_dHdq = super().dHdq(phase_space)
        # noML_dHdq shape is [nsamples, nparticle, DIM]

        print('dHdq_all q', phase_space.get_q())
        print('dHdq_all p', phase_space.get_p())
        print('dHdq_all tau', self.tau_cur)

        x = self.pack_dqdp_tau(phase_space, self.tau_cur)
        # x.shape = [nsamples * nparticle * nparticle, 5]

        predict = net(x)
        # predict.shape = [nsamples * nparticle * nparticle, 2]
        print('dHdq_all predict', predict)
        predict = self.unpack_dqdp_tau(predict, q_list.shape)
        # corrected_dHdq.shape = [nsamples, nparticle, DIM]

        corrected_dHdq = noML_dHdq + predict
        print('dHdq_all corrected_dHdq', corrected_dHdq)
        return corrected_dHdq

    # ===================================================
    def pack_dqdp_tau(self, phase_space, tau_cur):

        ''' function to prepare input in nn
        this function is use to make delta_q, delta_p, tau for input into model

        Parameters
        ----------
        tau_cur : float
                large time step for input in neural network
        phase_space : contains q_list, p_list as input for integration

        Returns
        ----------
        input in neural network
        here, 2*DIM + 1 is (del_qx, del_qy, del_px, del_py, tau )

        '''

        nsamples, nparticle, DIM = phase_space.get_q().shape

        print('pack_dqdp_tau q', phase_space.get_q())
        print('pack_dqdp_tau p', phase_space.get_p())

        dqdp_list = self.make_dqdp(phase_space)
        # dqdp_list.shape = [nsamples, nparticle, nparticle, 2*DIM]

        tau_tensor = torch.zeros([nsamples, nparticle, nparticle, 1], dtype=torch.float64)

        tau_tensor.fill_(tau_cur * 0.5)  # tau_tensor take them tau/2
        tau_diag_zero = torch.diagonal(tau_tensor, 0, 1, 2) # offset, nparticle, nparticle
        tau_diag_zero.fill_(0.0)   # tau_tensor make [:,i,i,1] element  zero

        x = torch.cat((dqdp_list, tau_tensor),dim = 3)
        # x.shape = [nsamples, nparticle, nparticle, 2*DIM + 1]

        x = torch.reshape(x,(nsamples*nparticle*nparticle,5))
        # x.shape = [ nsamples * nparticle * nparticle, 2*DIM + 1]
        print('pack_dqdp_tau x', x)

        return x

    # ===================================================

    def unpack_dqdp_tau(self, y, qlist_shape):
        ''' function to make output unpack

        parameter
        _____________
        y  : predict
                y.shape = [ nsamples * nparticle * nparticle, 2]

        return
        _____________
        y  : shape is  [nsamples,nparticle,2]
        '''

        nsamples, nparticle, DIM = qlist_shape
        print('unpack_dqdp_tau y', y)
        y = torch.reshape(y, (nsamples, nparticle, nparticle, DIM))
        print('unpack_dqdp_tau reshape y', y)
        # check - run "python3.7 -O" to switch off
        if __debug__:
            y2 = torch.clone(y)
            for i in range(nparticle): y2[:,i,i,:] = 0

        dy = torch.diagonal(y,0,1,2) # offset, nparticle, nparticle
        torch.fill_(dy,0.0)

        if __debug__:
            err = (y-y2)**2
            assert (torch.sum(err)<1e-6),'error in diagonal computations'

        print('unpack_dqdp_tau  y', y)
        y = torch.sum(y, dim=2) # sum over dim =2 that is all neighbors
        # y.shape = [nsamples,nparticle,2]
        print('unpack_dqdp_tau sum y', y)
        return y

    # ===================================================

    def make_dqdp(self, phase_space):

        ''' function to make dq and dp for feeding into nn

        Returns
        ----------
        take q_list and p_list, generate the difference matrix
        q_list.shape = p_list.shape = [nsamples, nparticle, DIM]
        dqdp : here 4 is (dq_x, dq_y, dp_x, dp_y )
        '''

        print('make_dqdp q', phase_space.get_q())
        print('make_dqdp p', phase_space.get_p())

        q_list = phase_space.get_q()
        p_list = phase_space.get_p()

        dq = self.delta_state(q_list)
        dp = self.delta_state(p_list)
        # dq.shape = dp.shape = [nsamples, nparticle, nparticle, 2]

        dqdp = torch.cat((dq, dp), dim=-1)
        # dqdp.shape is [ nsamples, nparticle, nparticle, 4]

        print('make_dqdp dqdp', dqdp)

        return dqdp

    # ===================================================

    def delta_state(self, state_list):

        ''' function to calculate distance of q or p between two particles

        Parameters
        ----------
        state_list : torch.tensor
                shape is [nsamples, nparticle, DIM]
        statem : repeated tensor which has the same shape as state0 along with dim=1
        statet : permutes the order of the axes of a tensor

        Returns
        ----------
        dstate : distance of q or p btw two particles
        '''

        state_len = state_list.shape[1]  # nparticle
        state0 = torch.unsqueeze(state_list, dim=1)
        # shape is [nsamples, 1, nparticle, DIM]

        statem = torch.repeat_interleave(state0, state_len, dim=1)
        # shape is [nsamples, nparticle, nparticle, DIM]

        statet = statem.permute(get_paired_distance_indices.permute_order)
        # shape is [nsamples, nparticle, nparticle, DIM]

        dstate = statet - statem
        # shape is [nsamples, nparticle, nparticle, DIM]

        return dstate

